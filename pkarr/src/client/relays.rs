use std::collections::HashMap;
use std::fmt::Debug;
#[cfg(not(wasm_browser))]
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_buffered::FuturesUnorderedBounded;
#[cfg(not(wasm_browser))]
use futures_lite::Stream;
use futures_lite::StreamExt;
use ntimestamp::Timestamp;
use url::Url;

use super::{ConcurrencyError, PublishError, QueryError};
use crate::relay_client::{RelayClient, RelayError};
use crate::{PublicKey, SignedPacket};

#[derive(Clone)]
pub struct RelaysClient {
    relays: Box<[RelayClient]>,
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
                .map(|relay| relay.base_url().as_str())
                .collect::<Vec<_>>(),
        );

        debug_struct.finish()
    }
}

impl RelaysClient {
    pub fn new(relays: Box<[Url]>, timeout: Duration) -> Result<Self, RelayError> {
        let inflight_publish = InflightPublishRequests::new(relays.len());
        let resolve_timeout = timeout;
        // Publish combines HTTP latency with the relay-side DHT PUT query.
        let publish_timeout = resolve_timeout
            .checked_mul(3)
            .ok_or_else(|| RelayError::Build("publish timeout overflow".to_string()))?;
        let relays = relays
            .into_vec()
            .into_iter()
            .map(|url| RelayClient::new(url, resolve_timeout, publish_timeout))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();

        Ok(Self {
            relays,
            inflight_publish,
        })
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

        for relay in &self.relays {
            let relay = relay.clone();
            let signed_packet = signed_packet.clone();
            let public_key = public_key.clone();
            let mut inflight = self.inflight_publish.clone();

            futures.push(async move {
                let result = relay
                    .publish(&signed_packet, cas)
                    .await
                    .map_err(map_relay_error);

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
            .expect("relays inflight publish requests done with no success or error!")
    }

    #[cfg(not(wasm_browser))]
    /// Cancel an inflight publish request.
    pub fn cancel_publish(&self, public_key: &PublicKey) {
        self.inflight_publish.cancel_request(public_key);
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

        self.relays.iter().for_each(|relay| {
            let relay = relay.clone();
            let public_key = public_key.clone();

            futures.push(async move {
                match relay
                    .resolve(
                        &public_key,
                        crate::ResolvePolicy::CacheFirst,
                        more_recent_than,
                    )
                    .await
                {
                    Ok(signed_packet) => signed_packet,
                    Err(error) => {
                        cross_debug!("GET {} {:?}", relay.base_url(), error);
                        None
                    }
                }
            });
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
    requests: Arc<Mutex<HashMap<PublicKey, InflightPublishRequest>>>,
}

impl InflightPublishRequests {
    fn new(relays_count: usize) -> Self {
        Self {
            relays_count,
            requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start_request(
        &self,
        public_key: &PublicKey,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let mut requests = self.requests.lock().expect("InflightPublishRequests lock");

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

    #[cfg(not(wasm_browser))]
    pub fn cancel_request(&self, public_key: &PublicKey) {
        let mut inflight = self.requests.lock().expect("InflightPublishRequests lock");

        inflight.remove(public_key);
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

    /// Returns true if request is done.
    fn add_success(&self, public_key: &PublicKey) -> Result<bool, PublishError> {
        let mut inflight = self.requests.lock().expect("InflightPublishRequests lock");

        if let Some(request) = inflight.get_mut(public_key) {
            let majority = self.relays_count / 2 + self.relays_count % 2;

            request.success_count += 1;

            if self.done(request) || request.success_count >= majority {
                inflight.remove(public_key);

                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(true)
        }
    }

    /// Returns true if request is done.
    fn add_error(
        &mut self,
        public_key: &PublicKey,
        error: PublishError,
    ) -> Result<bool, PublishError> {
        let mut inflight = self.requests.lock().expect("InflightPublishRequests lock");

        if let Some(request) = inflight.get_mut(public_key) {
            let majority = self.relays_count / 2 + self.relays_count % 2;

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
        } else {
            Ok(true)
        }
    }

    fn done(&self, request: &InflightPublishRequest) -> bool {
        let total_errors: usize = request.errors.values().sum();
        (total_errors + request.success_count) >= self.relays_count
    }
}

fn map_relay_error(error: RelayError) -> PublishError {
    match error {
        RelayError::Timeout | RelayError::DhtUnavailable => {
            PublishError::Query(QueryError::Timeout)
        }
        RelayError::BadRequest => {
            // This should be very unlikely unless relays are misbehaving, still worth
            // returning to the user to know that relays are misbehaving, and not just a
            // network issue.
            PublishError::Query(QueryError::BadRequest)
        }
        RelayError::NotMostRecent => PublishError::Concurrency(ConcurrencyError::NotMostRecent),
        RelayError::CasFailed => PublishError::Concurrency(ConcurrencyError::CasFailed),
        RelayError::ConflictRisk => PublishError::Concurrency(ConcurrencyError::ConflictRisk),
        RelayError::Build(_)
        | RelayError::Request(_)
        | RelayError::BodyTooLarge { .. }
        | RelayError::InvalidSignedPacket(_)
        | RelayError::InvalidSignedPacketSeq { .. }
        | RelayError::InvalidSignedPacketSeqHeader
        | RelayError::InvalidHeader(_)
        | RelayError::UnexpectedStatus(_) => PublishError::UnexpectedResponses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dht_relay_errors_map_to_publish_errors() {
        assert!(matches!(
            map_relay_error(RelayError::DhtUnavailable),
            PublishError::Query(QueryError::Timeout)
        ));
    }
}
