use std::collections::HashMap;
use std::fmt::Debug;
#[cfg(not(wasm_browser))]
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures_buffered::FuturesUnorderedBounded;
#[cfg(not(wasm_browser))]
use futures_lite::Stream;
use futures_lite::StreamExt;
use pubky_timestamp::Timestamp;
use reqwest::Method;
use url::Url;

use reqwest::{
    header::{self, HeaderValue},
    Client, StatusCode,
};

use super::{ConcurrencyError, PublishError, QueryError};
use crate::{PublicKey, SignedPacket};

pub struct RelaysClient {
    relays: Box<[Url]>,
    http_client: Client,
    timeout: Duration,
    pub(crate) inflight_publish: InflightPublishRequests,
}

impl Debug for RelaysClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("RelaysClient");

        debug_struct.field(
            "relays",
            &self
                .relays
                .as_ref()
                .iter()
                .map(|url| url.as_str())
                .collect::<Vec<_>>(),
        );

        debug_struct.finish()
    }
}

impl RelaysClient {
    pub fn new(relays: Box<[Url]>, timeout: Duration) -> Self {
        let inflight_publish = InflightPublishRequests::new(relays.len());

        Self {
            relays,
            http_client: Client::builder()
                .build()
                .expect("Client building should be infallible"),

            timeout,
            inflight_publish,
        }
    }

    pub async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let public_key = signed_packet.public_key();

        self.inflight_publish
            .start_request(&public_key, signed_packet, cas)?;

        let mut futures = futures_buffered::FuturesUnorderedBounded::new(self.relays.len());

        let body = signed_packet.to_relay_payload();
        let cas = cas.map(|timestamp| timestamp.format_http_date());

        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let timeout = self.timeout;

            let cas = cas.clone();
            let body = body.clone();

            let public_key = public_key.clone();
            let relay = relay.clone();

            let mut inflight = self.inflight_publish.clone();

            futures.push(async move {
                let result = publish_to_relay(http_client, relay, &public_key, body, cas, timeout)
                    .await
                    .map_err(map_reqwest_error);

                inflight.add_result(&public_key, result)
            });
        }

        futures
            .filter_map(|result| match result {
                Ok(true) => Some(Ok(())),
                Ok(false) => None,
                Err(err) => Some(Err(err)),
            })
            .next()
            .await
            .expect("infallible")
    }

    #[cfg(not(wasm_browser))]
    pub fn resolve(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> Pin<Box<dyn Stream<Item = SignedPacket> + Send>> {
        Box::pin(
            self.resolve_futures(public_key, more_recent_than)
                .filter_map(|opt| opt),
        )
    }

    pub fn resolve_futures(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> FuturesUnorderedBounded<impl futures_lite::Future<Output = Option<SignedPacket>>> {
        let mut futures = FuturesUnorderedBounded::new(self.relays.len());

        let if_modified_since = more_recent_than.map(|t| t.format_http_date());

        self.relays.iter().for_each(|relay| {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let public_key = public_key.clone();
            let if_modified_since = if_modified_since.clone();
            let timeout = self.timeout;

            futures.push(resolve_from_relay(
                http_client,
                relay,
                public_key,
                if_modified_since,
                timeout,
            ));
        });

        futures
    }
}

#[derive(Debug)]
struct InflightPublishRequest {
    signed_packet: SignedPacket,
    success_count: usize,
    errors: HashMap<PublishError, usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct InflightPublishRequests {
    relays_count: usize,
    requests: Arc<RwLock<HashMap<PublicKey, InflightPublishRequest>>>,
}

impl InflightPublishRequests {
    fn new(relays_count: usize) -> Self {
        Self {
            relays_count,
            requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn start_request(
        &self,
        public_key: &PublicKey,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let mut requests = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        if let Some(inflight_request) = requests.get_mut(public_key) {
            if signed_packet.signature() == inflight_request.signed_packet.signature() {
                // No-op, the inflight query is sufficient.
            } else if !signed_packet.more_recent_than(&inflight_request.signed_packet) {
                return Err(ConcurrencyError::NotMostRecent)?;
            } else if let Some(cas) = cas {
                if cas != inflight_request.signed_packet.timestamp() {
                    return Err(ConcurrencyError::CasFailed)?;
                }
            } else {
                return Err(ConcurrencyError::ConflictRisk)?;
            };
        } else {
            requests.insert(
                public_key.clone(),
                InflightPublishRequest {
                    signed_packet: signed_packet.clone(),
                    success_count: 0,
                    errors: Default::default(),
                },
            );
        };

        Ok(())
    }

    pub fn add_result(
        &mut self,
        public_key: &PublicKey,
        result: Result<(), PublishError>,
    ) -> Result<bool, PublishError> {
        match result {
            Ok(_) => self.add_success(public_key),
            Err(error) => self.add_error(public_key, error),
        }
    }

    fn add_success(&self, public_key: &PublicKey) -> Result<bool, PublishError> {
        let mut inflight = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let request = inflight.get_mut(public_key).expect("infallible");
        let majority = (self.relays_count / 2) + 1;

        request.success_count += 1;

        if self.done(request) {
            return Ok(true);
        } else if request.success_count >= majority {
            inflight.remove(public_key);

            return Ok(true);
        }

        Ok(false)
    }

    fn add_error(
        &mut self,
        public_key: &PublicKey,
        error: PublishError,
    ) -> Result<bool, PublishError> {
        let mut inflight = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let request = inflight.get_mut(public_key).expect("infallible");
        let majority = (self.relays_count / 2) + 1;

        // Add error, and return early error if necessary.
        {
            let count = request.errors.get(&error).unwrap_or(&0) + 1;

            if count >= majority
                && matches!(
                    error,
                    PublishError::Concurrency(ConcurrencyError::NotMostRecent)
                ) | matches!(
                    error,
                    PublishError::Concurrency(ConcurrencyError::CasFailed)
                )
            {
                inflight.remove(public_key);

                return Err(error);
            }

            request.errors.insert(error, count);
        }

        if self.done(request) {
            let request = inflight.remove(public_key).expect("infallible");

            if request.success_count >= majority {
                Ok(true)
            } else {
                let most_common_error = request
                    .errors
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(error, _)| error)
                    .expect("infallible");

                Err(most_common_error)
            }
        } else {
            Ok(false)
        }
    }

    fn done(&self, request: &InflightPublishRequest) -> bool {
        (request.errors.len() + request.success_count) == self.relays_count
    }
}

pub async fn publish_to_relay(
    http_client: reqwest::Client,
    relay: Url,
    public_key: &PublicKey,
    body: Bytes,
    cas: Option<String>,
    timeout: Duration,
) -> Result<(), reqwest::Error> {
    let url = format_url(&relay, public_key);

    let mut request = http_client.put(url.clone());

    // TODO: enable for wasm after reqwest release
    #[cfg(not(wasm_browser))]
    {
        request = request
            // Publish combines the http latency with the PUT query to the dht
            // on the relay side, so we should be as generous as possible
            .timeout(timeout * 3);
    }

    if let Some(date) = cas {
        request = request.header(header::IF_UNMODIFIED_SINCE, date);
    }

    let response = request.body(body).send().await.inspect_err(|error| {
        cross_debug!("PUT {:?}", error);
    })?;

    let status = response.status();

    if let Err(error) = response.error_for_status_ref() {
        let text = response.text().await.unwrap_or("".to_string());

        cross_debug!("Got error response for PUT {url} {status} {text}");

        return Err(error);
    };

    if status.is_success() {
        cross_debug!("Successfully published to {url}");
    } else {
        cross_debug!("Got neither 2xx nor >=400 status code {status} for PUT {url}",);
    }

    Ok(())
}

fn map_reqwest_error(error: reqwest::Error) -> PublishError {
    if error.is_timeout() {
        PublishError::Query(QueryError::Timeout)
    } else if error.is_status() {
        match error
            .status()
            .expect("previously verified that it is a status error")
        {
            StatusCode::BAD_REQUEST => {
                // This should be very unlikely unless relays are misbehaving, still worth
                // returning to the user to know that relays are misbehaving, and not just a
                // network issue.
                PublishError::Query(QueryError::BadRequest)
            }
            StatusCode::CONFLICT => PublishError::Concurrency(ConcurrencyError::NotMostRecent),
            StatusCode::PRECONDITION_FAILED => {
                PublishError::Concurrency(ConcurrencyError::CasFailed)
            }
            StatusCode::PRECONDITION_REQUIRED => {
                PublishError::Concurrency(ConcurrencyError::ConflictRisk)
            }
            _ => PublishError::UnexpectedResponses,
        }
    } else {
        PublishError::UnexpectedResponses
    }
}

pub fn format_url(relay: &Url, public_key: &PublicKey) -> Url {
    let mut url = relay.clone();

    let mut segments = url
        .path_segments_mut()
        .expect("Relay url cannot be base, is it http(s)?");

    segments.push(&public_key.to_string());

    drop(segments);

    url
}

pub async fn resolve_from_relay(
    http_client: reqwest::Client,
    relay: Url,
    public_key: PublicKey,
    if_modified_since: Option<String>,
    timeout: Duration,
) -> Option<SignedPacket> {
    let url = format_url(&relay, &public_key);

    let mut request = reqwest::Request::new(Method::GET, url.clone());

    #[cfg(not(wasm_browser))]
    {
        *request.timeout_mut() = Some(timeout);
    }

    if let Some(ref httpdate) = if_modified_since {
        request.headers_mut().insert(
            header::IF_MODIFIED_SINCE,
            HeaderValue::from_str(httpdate).expect("httpdate to be valid header value"),
        );
    }

    let response = loop {
        let response = match http_client
            .execute(request.try_clone().expect("infallible"))
            .await
        {
            Ok(response) => response,
            Err(error) => {
                cross_debug!("GET {:?}", error);

                return None;
            }
        };

        let status = response.status();

        if response.error_for_status_ref().is_err() {
            let text = response.text().await.unwrap_or("".to_string());

            cross_debug!("Got error response for GET {url} {status} {text}");

            return None;
        };

        if should_retry_with_cache_disabled(
            &mut request,
            &if_modified_since,
            &status,
            response.content_length(),
        ) {
            continue;
        } else {
            break response;
        }
    };

    if response.content_length().unwrap_or_default() > SignedPacket::MAX_BYTES {
        cross_debug!("Response too large for GET {url}");

        return None;
    }

    let payload = match response.bytes().await {
        Ok(payload) => payload,
        Err(error) => {
            cross_debug!("Failed to read relay response from GET {url} {error}");

            return None;
        }
    };

    match SignedPacket::from_relay_payload(&public_key, &payload) {
        Ok(signed_packet) => Some(signed_packet),
        Err(error) => {
            cross_debug!("Invalid signed_packet {url}:{error}");

            None
        }
    }
}

fn should_retry_with_cache_disabled(
    request: &mut reqwest::Request,
    if_modified_since: &Option<String>,
    status: &StatusCode,
    content_length: Option<u64>,
) -> bool {
    // We got a 304 not modified, even though we don't have any packet in cache.
    let unexpected_not_modified =
        (*status == StatusCode::NOT_MODIFIED) && if_modified_since.is_none();

    // We got a success response, but the packet is empty.
    let broken_cache = status.is_success() && (content_length.unwrap_or_default() == 0);

    let needs_retry_with_cache_disabled = unexpected_not_modified || broken_cache;

    let havent_retried_with_cache_disabled_already =
        request.headers().get(header::CACHE_CONTROL).is_none();

    if needs_retry_with_cache_disabled && havent_retried_with_cache_disabled_already {
        request.headers_mut().insert(
            header::CACHE_CONTROL,
            "no-cache, no-store, must-revalidate"
                .try_into()
                .expect("cache control is valid http header value"),
        );

        return true;
    }

    false
}
