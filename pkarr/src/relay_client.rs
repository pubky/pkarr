//! Single-relay HTTP client.

use bytes::Bytes;
use ntimestamp::Timestamp;
use reqwest::{header, Client, Response, StatusCode};
use std::time::Duration;
use url::Url;

use crate::{
    PublicKey, ResolvePolicy, SignedPacket, StoredNodeCount, PKARR_DHT_STORED_NODES,
    PKARR_INVALID_SIGNED_PACKET_SEQ,
};

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
    timeout: Duration,
}

impl RelayClient {
    /// Build a client for one relay endpoint using a [`reqwest::Client`].
    ///
    /// The timeout is applied to each relay request individually. This
    /// keeps timeout behavior available on WASM targets, where reqwest does not
    /// expose client-wide timeout configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `base_url` is not a valid relay base URL.
    pub fn new(base_url: Url, client: Client, timeout: Duration) -> Result<Self, RelayError> {
        if base_url.cannot_be_a_base() {
            return Err(RelayError::Build(format!(
                "Invalid base url of a relay: `{base_url}`"
            )));
        }

        Ok(Self {
            base_url,
            client,
            timeout,
        })
    }

    /// Return this client's relay base URL.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Publish a [`SignedPacket`] to this relay.
    ///
    /// # Returns
    ///
    /// Returns a [`StoredNodeCount`] with the number of DHT nodes that
    /// acknowledged storing the packet. Older relays that do not return this
    /// count are treated as if one DHT node acknowledged storing the packet.
    ///
    /// # Errors
    ///
    /// Returns an error when the relay request fails, the relay returns a
    /// non-success status, the request times out, or the relay returns a
    /// malformed stored-node count header.
    pub async fn publish(&self, packet: &SignedPacket) -> Result<StoredNodeCount, RelayError> {
        let url = self.build_url(&packet.public_key(), None);

        let response = self
            .client
            .put(url.clone())
            .timeout(self.timeout)
            .body(packet.to_relay_payload())
            .send()
            .await
            .map_err(RelayError::from_reqwest)?;
        let status = response.status();

        if status.is_success() {
            let stored_on = extract_dht_stored_nodes(&response)?;
            debug!("Successfully published to {url}");
            return Ok(stored_on);
        }

        let text = response.text().await.unwrap_or_default();
        debug!("Got error response for PUT {url} {status} {text}");

        Err(RelayError::from_status(status))
    }

    /// Resolve a [`SignedPacket`] from this relay.
    ///
    /// [`ResolvePolicy::DhtNetworkOnly`] bypasses HTTP caches on the initial
    /// request so the relay queries current DHT network state.
    ///
    /// # Errors
    ///
    /// Returns an error when the relay request fails, the response body is too
    /// large, the relay has no matching packet, the returned relay payload does
    /// not verify, or the relay reports a newer DHT mutable item that is not a
    /// valid signed packet.
    pub async fn resolve(
        &self,
        key: &PublicKey,
        policy: ResolvePolicy,
        newer_than: Option<Timestamp>,
    ) -> Result<SignedPacket, RelayError> {
        let url = self.build_url(key, Some(policy));
        let bypass_cache = policy == ResolvePolicy::DhtNetworkOnly;

        let mut response = self
            .send_resolve_request(&url, bypass_cache, newer_than)
            .await?;

        // A stale HTTP cache can produce impossible 304 responses or empty
        // successful responses. Retry once with cache bypass headers.
        if should_retry_with_cache_bypass(&response, newer_than.as_ref()) {
            response = self.send_resolve_request(&url, true, newer_than).await?;
        }

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            if let Some(seq) = extract_invalid_signed_packet_seq(&response)? {
                if invalid_seq_is_not_newer_than_request(seq, newer_than.as_ref()) {
                    return Err(RelayError::NotFound);
                }

                return Err(RelayError::InvalidSignedPacketSeq { seq });
            }

            return Err(RelayError::NotFound);
        }

        if status == StatusCode::NOT_MODIFIED {
            return Err(RelayError::NotFound);
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

        if newer_than.is_some_and(|newer_than| signed_packet.timestamp() <= newer_than) {
            return Err(RelayError::NotFound);
        }

        Ok(signed_packet)
    }

    async fn send_resolve_request(
        &self,
        url: &Url,
        bypass_cache: bool,
        newer_than: Option<Timestamp>,
    ) -> Result<Response, RelayError> {
        let mut request = self.client.get(url.clone()).timeout(self.timeout);

        if bypass_cache {
            request = request.header(header::CACHE_CONTROL, CACHE_BYPASS);
        }

        if let Some(newer_than) = newer_than {
            request = request.header(header::IF_MODIFIED_SINCE, newer_than.format_http_date());
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

    /// Relay returned an invalid signed packet payload.
    #[error("relay returned an invalid signed packet: {0}")]
    InvalidSignedPacket(crate::errors::SignedPacketVerifyError),

    /// Relay returned a malformed invalid-signed-packet sequence header.
    #[error("relay returned a malformed Pkarr-Invalid-Signed-Packet-Seq header")]
    InvalidSignedPacketSeqHeader,

    /// Relay returned a malformed DHT stored nodes header.
    #[error("relay returned a malformed Pkarr-Dht-Stored-Nodes header")]
    InvalidDhtStoredNodesHeader,

    /// Relay request contained an invalid header.
    #[error("relay request contained an invalid header: {0}")]
    InvalidHeader(header::InvalidHeaderValue),

    /// Relay returned bad request.
    #[error("relay returned bad request")]
    BadRequest,

    /// Relay has a more recent signed packet.
    #[error("relay has a more recent signed packet")]
    NotMostRecent,

    /// Relay could not complete a DHT query.
    #[error("relay DHT query is unavailable")]
    DhtUnavailable,

    /// Relay returned an unexpected status code.
    #[error("relay returned unexpected status code {0}")]
    UnexpectedStatus(StatusCode),

    /// Relay reported a newer DHT mutable item that is not a valid signed packet.
    #[error(
        "relay reported a newer DHT mutable item at seq {seq} that is not a valid signed packet"
    )]
    InvalidSignedPacketSeq {
        /// Mutable item sequence number.
        seq: i64,
    },

    /// Relay found no signed packet.
    #[error("relay found no signed packet")]
    NotFound,
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
            StatusCode::CONFLICT | StatusCode::PRECONDITION_REQUIRED => Self::NotMostRecent,
            StatusCode::SERVICE_UNAVAILABLE => Self::DhtUnavailable,
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

fn extract_invalid_signed_packet_seq(response: &Response) -> Result<Option<i64>, RelayError> {
    response
        .headers()
        .get(PKARR_INVALID_SIGNED_PACKET_SEQ)
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|value| value.parse().ok())
                .ok_or(RelayError::InvalidSignedPacketSeqHeader)
        })
        .transpose()
}

fn extract_dht_stored_nodes(response: &Response) -> Result<StoredNodeCount, RelayError> {
    let Some(value) = response.headers().get(PKARR_DHT_STORED_NODES) else {
        return Ok(1);
    };

    value
        .to_str()
        .ok()
        .and_then(|value| value.parse().ok())
        .ok_or(RelayError::InvalidDhtStoredNodesHeader)
}

fn invalid_seq_is_not_newer_than_request(seq: i64, newer_than: Option<&Timestamp>) -> bool {
    let Some(newer_than) = newer_than else {
        return false;
    };

    seq >= 0 && seq as u64 <= newer_than.as_u64()
}

fn should_retry_with_cache_bypass(
    response: &Response,
    if_modified_since: Option<&Timestamp>,
) -> bool {
    let status = response.status();
    let unexpected_not_modified = status == StatusCode::NOT_MODIFIED && if_modified_since.is_none();
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
        Arc, Mutex,
    };
    use std::time::Duration;

    use axum::{
        http::{HeaderMap, StatusCode as HttpStatusCode},
        routing::{get, put},
        Router,
    };
    use ntimestamp::Timestamp;
    use rstest::rstest;
    use tokio::net::TcpListener;
    use url::Url;

    use super::*;
    use crate::Keypair;

    const TIMEOUT: Duration = Duration::from_secs(1);

    fn test_client(base_url: Url) -> RelayClient {
        let client = Client::builder().build().unwrap();
        RelayClient::new(base_url, client, TIMEOUT).unwrap()
    }

    #[test]
    fn relay_503_status_maps_to_dht_unavailable() {
        assert!(matches!(
            RelayError::from_status(StatusCode::SERVICE_UNAVAILABLE),
            RelayError::DhtUnavailable
        ));
    }

    #[tokio::test]
    async fn publish_returns_dht_stored_nodes() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            put(|| async { (HttpStatusCode::NO_CONTENT, [(PKARR_DHT_STORED_NODES, "7")]) }),
        );
        let client = test_client(serve(app).await);

        let stored_on = client.publish(&signed_packet).await.unwrap();

        assert_eq!(stored_on, 7);
    }

    #[tokio::test]
    async fn publish_rejects_malformed_dht_stored_nodes_header() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            put(|| async {
                (
                    HttpStatusCode::NO_CONTENT,
                    [(PKARR_DHT_STORED_NODES, "not a count")],
                )
            }),
        );
        let client = test_client(serve(app).await);

        let error = client.publish(&signed_packet).await.unwrap_err();

        assert!(matches!(error, RelayError::InvalidDhtStoredNodesHeader));
    }

    #[tokio::test]
    async fn publish_defaults_missing_dht_stored_nodes_header_to_one() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            put(|| async { HttpStatusCode::NO_CONTENT }),
        );
        let client = test_client(serve(app).await);

        let stored_on = client.publish(&signed_packet).await.unwrap();

        assert_eq!(stored_on, 1);
    }

    #[tokio::test]
    async fn resolve_maps_503_to_dht_unavailable() {
        assert!(matches!(
            resolve_error_for_status(HttpStatusCode::SERVICE_UNAVAILABLE).await,
            RelayError::DhtUnavailable
        ));
    }

    #[tokio::test]
    async fn resolve_returns_not_found_on_not_found() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { HttpStatusCode::NOT_FOUND }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::NotFound));
    }

    #[tokio::test]
    async fn resolve_returns_invalid_signed_packet_seq_on_not_found() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async {
                (
                    HttpStatusCode::NOT_FOUND,
                    [(PKARR_INVALID_SIGNED_PACKET_SEQ, "42")],
                )
            }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            RelayError::InvalidSignedPacketSeq { seq: 42 }
        ));
    }

    #[tokio::test]
    async fn resolve_maps_invalid_signed_packet_seq_at_request_bound_to_not_found() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async {
                (
                    HttpStatusCode::NOT_FOUND,
                    [(PKARR_INVALID_SIGNED_PACKET_SEQ, "41")],
                )
            }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::CacheFirst,
                Some(Timestamp::from(41)),
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::NotFound));
    }

    #[tokio::test]
    async fn resolve_reports_invalid_signed_packet_seq_newer_than_request_bound() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async {
                (
                    HttpStatusCode::NOT_FOUND,
                    [(PKARR_INVALID_SIGNED_PACKET_SEQ, "43")],
                )
            }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::CacheFirst,
                Some(Timestamp::from(41)),
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            RelayError::InvalidSignedPacketSeq { seq: 43 }
        ));
    }

    #[tokio::test]
    async fn resolve_rejects_malformed_invalid_signed_packet_seq_header() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async {
                (
                    HttpStatusCode::NOT_FOUND,
                    [(PKARR_INVALID_SIGNED_PACKET_SEQ, "not a seq")],
                )
            }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::InvalidSignedPacketSeqHeader));
    }

    #[tokio::test]
    async fn resolve_returns_not_found_on_not_modified() {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(|| async { HttpStatusCode::NOT_MODIFIED }),
        );
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                Some(Timestamp::now()),
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::NotFound));
    }

    #[tokio::test]
    async fn dht_network_only_resolve_bypasses_http_cache() {
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
                            .and_then(|value| value.to_str().ok())
                        {
                            Some(CACHE_BYPASS) => (HttpStatusCode::OK, payload),
                            _ => (HttpStatusCode::BAD_REQUEST, Bytes::new()),
                        }
                    }
                }
            }),
        );
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(&keypair.public_key(), ResolvePolicy::DhtNetworkOnly, None)
            .await
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
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
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[rstest]
    #[case::nonzero(Timestamp::from(42))]
    #[case::zero(Timestamp::from(0))]
    #[tokio::test]
    async fn resolve_sends_if_modified_since(#[case] newer_than: Timestamp) {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
        let expected_header = newer_than.format_http_date();
        let seen_header = Arc::new(Mutex::new(None));
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get({
                let payload = signed_packet.to_relay_payload();
                let seen_header = seen_header.clone();

                move |headers: HeaderMap| {
                    let payload = payload.clone();
                    let seen_header = seen_header.clone();

                    async move {
                        let if_modified_since = headers
                            .get(header::IF_MODIFIED_SINCE)
                            .and_then(|value| value.to_str().ok())
                            .map(ToOwned::to_owned);
                        *seen_header.lock().unwrap() = if_modified_since;

                        payload
                    }
                }
            }),
        );
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::CacheFirst,
                Some(newer_than),
            )
            .await
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
        assert_eq!(
            seen_header.lock().unwrap().as_deref(),
            Some(expected_header.as_str())
        );
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
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
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
        let client = test_client(serve(app).await);

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
        let client = test_client(serve(app).await);

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

        let client = test_client(base_url);
        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
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
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
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
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
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
        let client = test_client(serve(app).await);

        let resolved = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[tokio::test]
    async fn resolve_rejects_packet_with_same_timestamp_as_newer_than() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .timestamp(Timestamp::from(42))
            .sign(&keypair)
            .unwrap();
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
        let client = test_client(serve(app).await);

        let error = client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::CacheFirst,
                Some(signed_packet.timestamp()),
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RelayError::NotFound));
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

        let client = test_client(https_url);
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

    async fn resolve_error_for_status(status: HttpStatusCode) -> RelayError {
        let keypair = Keypair::random();
        let app = Router::new().route(
            &format!("/{}", keypair.public_key()),
            get(move || async move { status }),
        );
        let client = test_client(serve(app).await);

        client
            .resolve(
                &keypair.public_key(),
                ResolvePolicy::LocalOrRelayCacheOnly,
                None,
            )
            .await
            .unwrap_err()
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
