//! Single-relay HTTP client.

use bytes::Bytes;
use ntimestamp::Timestamp;
use reqwest::{
    header::{self, HeaderValue},
    Client, Response, StatusCode,
};
use std::time::Duration;
use url::Url;

use crate::{PublicKey, ResolvePolicy, SignedPacket};

const MEMENTO_DATETIME: &str = "memento-datetime";
const CACHE_BYPASS: &str = "no-cache, no-store, must-revalidate";

macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::debug!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::debug!($($arg)*);
    };
}

/// Single-relay HTTP client.
#[derive(Clone, Debug)]
pub struct RelayClient {
    base_url: Url,
    client: Client,
    resolve_timeout: Duration,
    publish_timeout: Duration,
}

impl RelayClient {
    /// Build a client for one relay endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn new(
        base_url: Url,
        resolve_timeout: Duration,
        publish_timeout: Duration,
    ) -> Result<Self, RelayError> {
        let client = Client::builder()
            .build()
            .map_err(|e| RelayError::Build(e.to_string()))?;
        if base_url.cannot_be_a_base() {
            return Err(RelayError::Build(format!(
                "Invalid base url of a relay: `{base_url}`"
            )));
        }

        Ok(Self {
            base_url,
            client,
            resolve_timeout,
            publish_timeout,
        })
    }

    /// Return this client's relay base URL.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Publish a [`SignedPacket`] to this relay.
    ///
    /// # Errors
    ///
    /// Returns an error when the relay request fails, the relay returns a
    /// non-success status, or the request times out.
    pub async fn publish(
        &self,
        packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), RelayError> {
        let url = self.build_url(&packet.public_key(), None);

        let mut request = self
            .client
            .put(url.clone())
            .timeout(self.publish_timeout)
            .body(packet.to_relay_payload());

        if let Some(cas) = cas {
            request = request.header(header::IF_MATCH, cas.as_u64().to_string());
        }

        let response = request.send().await.map_err(RelayError::from_reqwest)?;
        let status = response.status();

        if status.is_success() {
            debug!("Successfully published to {url}");
            return Ok(());
        }

        let text = response.text().await.unwrap_or_default();
        debug!("Got error response for PUT {url} {status} {text}");

        Err(RelayError::from_status(status))
    }

    /// Resolve a [`SignedPacket`] from this relay.
    ///
    /// # Errors
    ///
    /// Returns an error when the relay request fails, the response body is too
    /// large, or the returned relay payload does not verify.
    pub async fn resolve(
        &self,
        key: &PublicKey,
        policy: ResolvePolicy,
        newer_than: Option<Timestamp>,
    ) -> Result<Option<SignedPacket>, RelayError> {
        let url = self.build_url(key, Some(policy));

        let mut response = self
            .send_resolve_request(&url, newer_than.as_ref(), false)
            .await?;

        // A stale HTTP cache can produce impossible 304 responses or empty
        // successful responses. Retry once with cache bypass headers.
        if should_retry_with_cache_bypass(&response, newer_than.as_ref()) {
            response = self
                .send_resolve_request(&url, newer_than.as_ref(), true)
                .await?;
        }

        let status = response.status();
        if status == StatusCode::NOT_MODIFIED || status == StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if status.is_client_error() || status.is_server_error() {
            let text = response.text().await.unwrap_or_default();
            debug!("Got error response for GET {url} {status} {text}");

            return Err(RelayError::from_status(status));
        }

        let last_seen = extract_memento_datetime(&response);
        let payload = read_body_with_limit(response, SignedPacket::MAX_BYTES as usize).await?;

        let mut signed_packet = SignedPacket::from_relay_payload(key, &payload)
            .map_err(RelayError::InvalidSignedPacket)?;

        if let Some(last_seen) = last_seen {
            signed_packet.set_last_seen(&last_seen);
        }

        Ok(Some(signed_packet))
    }

    async fn send_resolve_request(
        &self,
        url: &Url,
        newer_than: Option<&Timestamp>,
        bypass_cache: bool,
    ) -> Result<Response, RelayError> {
        let mut request = self.client.get(url.clone()).timeout(self.resolve_timeout);

        if let Some(newer_than) = newer_than {
            let newer_than = newer_than.format_http_date();
            request = request.header(
                header::IF_MODIFIED_SINCE,
                HeaderValue::from_str(&newer_than).map_err(RelayError::InvalidHeader)?,
            );
        }

        if bypass_cache {
            request = request.header(header::CACHE_CONTROL, CACHE_BYPASS);
        }

        request.send().await.map_err(RelayError::from_reqwest)
    }

    fn build_url(&self, public_key: &PublicKey, policy: Option<ResolvePolicy>) -> Url {
        let mut url = self.base_url.clone();

        {
            let mut segments = url
                .path_segments_mut()
                .expect("relay URL cannot be a base URL");

            segments.push(&public_key.to_string());
        }

        if let Some(policy) = policy {
            url.query_pairs_mut().append_pair("policy", policy.as_str());
        }

        url
    }
}

/// Relay-client error.
#[derive(thiserror::Error, Debug)]
pub enum RelayError {
    /// Failed to build the relay client.
    #[error("failed to build relay client: {0}")]
    Build(String),

    /// Relay request timed out.
    #[error("relay request timed out")]
    Timeout,

    /// Relay request failed.
    #[error("relay request failed: {0}")]
    Request(reqwest::Error),

    /// Relay response body exceeded the maximum accepted size.
    #[error("relay response body is larger than {limit} bytes")]
    BodyTooLarge {
        /// Maximum accepted size.
        limit: usize,
    },

    /// Relay returned a malformed signed packet payload.
    #[error("relay returned an invalid signed packet: {0}")]
    InvalidSignedPacket(crate::errors::SignedPacketVerifyError),

    /// Relay request contained an invalid header.
    #[error("relay request contained an invalid header: {0}")]
    InvalidHeader(header::InvalidHeaderValue),

    /// Relay returned bad request.
    #[error("relay returned bad request")]
    BadRequest,

    /// Relay has a more recent signed packet.
    #[error("relay has a more recent signed packet")]
    NotMostRecent,

    /// Relay compare-and-swap failed.
    #[error("relay compare-and-swap failed")]
    CasFailed,

    /// Relay requires compare-and-swap.
    #[error("relay requires compare-and-swap")]
    ConflictRisk,

    /// Relay returned an unexpected status code.
    #[error("relay returned unexpected status code {0}")]
    UnexpectedStatus(StatusCode),
}

impl RelayError {
    fn from_reqwest(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout
        } else if let Some(status) = error.status() {
            Self::from_status(status)
        } else {
            Self::Request(error)
        }
    }

    fn from_status(status: StatusCode) -> Self {
        match status {
            StatusCode::BAD_REQUEST => Self::BadRequest,
            StatusCode::CONFLICT => Self::NotMostRecent,
            StatusCode::PRECONDITION_FAILED => Self::CasFailed,
            StatusCode::PRECONDITION_REQUIRED => Self::ConflictRisk,
            status => Self::UnexpectedStatus(status),
        }
    }
}

fn extract_memento_datetime(response: &Response) -> Option<Timestamp> {
    response
        .headers()
        .get(MEMENTO_DATETIME)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Timestamp::parse_http_date(value).ok())
        .filter(|timestamp| timestamp <= &Timestamp::now())
}

fn should_retry_with_cache_bypass(response: &Response, newer_than: Option<&Timestamp>) -> bool {
    let status = response.status();
    let unexpected_not_modified = status == StatusCode::NOT_MODIFIED && newer_than.is_none();
    let empty_success = status.is_success() && response.content_length().unwrap_or_default() == 0;

    unexpected_not_modified || empty_success
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_body_with_limit(mut response: Response, limit: usize) -> Result<Bytes, RelayError> {
    reject_known_too_large(response.content_length(), limit)?;

    let mut payload = bytes::BytesMut::with_capacity(limit);

    while let Some(chunk) = response.chunk().await.map_err(RelayError::from_reqwest)? {
        if payload.len() + chunk.len() > limit {
            return Err(RelayError::BodyTooLarge { limit });
        }

        payload.extend_from_slice(&chunk);
    }

    Ok(payload.freeze())
}

#[cfg(target_arch = "wasm32")]
async fn read_body_with_limit(response: Response, limit: usize) -> Result<Bytes, RelayError> {
    reject_known_too_large(response.content_length(), limit)?;

    let payload = response.bytes().await.map_err(RelayError::from_reqwest)?;

    if payload.len() > limit {
        return Err(RelayError::BodyTooLarge { limit });
    }
    Ok(payload)
}

fn reject_known_too_large(content_length: Option<u64>, limit: usize) -> Result<(), RelayError> {
    match content_length {
        Some(length) if length > limit as u64 => Err(RelayError::BodyTooLarge { limit }),
        _ => Ok(()),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use axum::{
        http::{HeaderMap, StatusCode as HttpStatusCode},
        routing::get,
        Router,
    };
    use ntimestamp::Timestamp;
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::Keypair;

    const TIMEOUT: Duration = Duration::from_secs(1);

    #[tokio::test]
    async fn resolve_returns_none_on_not_found() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { HttpStatusCode::NOT_FOUND }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap();

        assert!(resolved.is_none());
    }

    #[tokio::test]
    async fn resolve_returns_none_on_not_modified() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { HttpStatusCode::NOT_MODIFIED }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                Some(Timestamp::now()),
            )
            .await
            .unwrap();

        assert!(resolved.is_none());
    }

    #[tokio::test]
    async fn resolve_retries_unexpected_not_modified_with_cache_bypass() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();

                move |headers: HeaderMap| {
                    let payload = payload.clone();

                    async move {
                        match headers
                            .get(header::CACHE_CONTROL)
                            .and_then(|v| v.to_str().ok())
                        {
                            Some(CACHE_BYPASS) => (HttpStatusCode::OK, payload),
                            _ => (HttpStatusCode::NOT_MODIFIED, Bytes::new()),
                        }
                    }
                }
            }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[tokio::test]
    async fn resolve_retries_empty_success_with_cache_bypass() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();

                move |headers: HeaderMap| {
                    let payload = payload.clone();

                    async move {
                        match headers
                            .get(header::CACHE_CONTROL)
                            .and_then(|v| v.to_str().ok())
                        {
                            Some(CACHE_BYPASS) => payload,
                            _ => Bytes::new(),
                        }
                    }
                }
            }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[tokio::test]
    async fn resolve_rejects_response_body_larger_than_signed_packet_limit() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { vec![0; SignedPacket::MAX_BYTES as usize + 1] }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::BodyTooLarge { .. }));
    }

    #[test]
    fn rejects_known_too_large_content_length_before_buffering() {
        assert!(reject_known_too_large(Some(101), 100).is_err());
        assert!(reject_known_too_large(Some(100), 100).is_ok());
        assert!(reject_known_too_large(None, 100).is_ok());
    }

    #[tokio::test]
    async fn resolve_rejects_invalid_signed_packet_payload() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { vec![0; 1] }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::InvalidSignedPacket(_)));
    }

    #[tokio::test]
    async fn resolve_sets_last_seen_from_memento_datetime() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let last_seen = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();

        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();
                let last_seen = last_seen.format_http_date();

                move || {
                    let payload = payload.clone();
                    let last_seen = last_seen.clone();

                    async move { ([(MEMENTO_DATETIME, last_seen)], payload) }
                }
            }),
        );

        let base_url = serve(app).await;

        let client = RelayClient::new(base_url, TIMEOUT, TIMEOUT).unwrap();
        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
        assert_eq!(resolved.last_seen(), &last_seen);
    }

    #[tokio::test]
    async fn resolve_ignores_invalid_memento_datetime() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();

                move || {
                    let payload = payload.clone();

                    async move { ([(MEMENTO_DATETIME, "not an http date")], payload) }
                }
            }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
        assert_ne!(
            resolved.last_seen(),
            &Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap()
        );
    }

    #[tokio::test]
    async fn resolve_ignores_future_memento_datetime() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let future_last_seen = Timestamp::now() + 60_000_000;

        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();
                let future_last_seen = future_last_seen.format_http_date();

                move || {
                    let payload = payload.clone();
                    let future_last_seen = future_last_seen.clone();

                    async move { ([(MEMENTO_DATETIME, future_last_seen)], payload) }
                }
            }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
        assert!(resolved.last_seen() <= &Timestamp::now());
    }

    #[tokio::test]
    async fn resolve_succeeds_without_memento_datetime() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();

                move || {
                    let payload = payload.clone();

                    async move { payload }
                }
            }),
        );
        let client = RelayClient::new(serve(app).await, TIMEOUT, TIMEOUT).unwrap();

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    // We test that the client can work with HTTPS endpoints; however, this does
    // not guarantee that it will work in a release build because, in tests, we
    // also have dev dependencies.
    #[tokio::test]
    async fn client_works_with_https() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let https_url = format!("https://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let reached_listener = Arc::new(AtomicBool::new(false));

        tokio::spawn({
            let reached_listener = reached_listener.clone();
            async move {
                if listener.accept().await.is_ok() {
                    reached_listener.store(true, Ordering::SeqCst);
                }
            }
        });

        let client = RelayClient::new(https_url, TIMEOUT, TIMEOUT).unwrap();
        let keypair = Keypair::random();
        let err = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            reached_listener.load(Ordering::SeqCst),
            "HTTPS request never reached the TCP listener, got {err:?}"
        );
    }

    async fn serve(app: Router) -> Url {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        base_url
    }
}
