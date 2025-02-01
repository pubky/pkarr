use std::collections::HashMap;
use std::fmt::Debug;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use flume::Sender;
use futures_buffered::FuturesUnorderedBounded;
use futures_lite::{Stream, StreamExt};
use pubky_timestamp::Timestamp;
use reqwest::{Client, StatusCode};
use url::Url;

use crate::{PublicKey, SignedPacket};

use crate::client::shared::{publish_to_relay, resolve_from_relay};

use super::native::{ConcurrencyError, PublishError, QueryError};

pub struct RelaysClient {
    relays: Box<[Url]>,
    http_client: Client,
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
                .timeout(timeout)
                .build()
                .expect("Client building should be infallible"),

            inflight_publish,
        }
    }

    pub fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> flume::Receiver<Result<(), PublishError>> {
        let (tx, rx) = flume::bounded(self.relays.len());

        let public_key = signed_packet.public_key();
        let body = signed_packet.to_relay_payload();

        if let Err(error) =
            self.inflight_publish
                .add_sender(&public_key, tx.clone(), signed_packet, cas)
        {
            let _ = tx.send(Err(error));
        };

        let cas = cas.map(|timestamp| timestamp.format_http_date());

        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let public_key = public_key.clone();
            let body = body.clone();
            let cas = cas.clone();
            let inflight = self.inflight_publish.clone();

            tokio::spawn(async move {
                let result = publish_to_relay(http_client, relay, &public_key, body, cas)
                    .await
                    .map_err(map_reqwest_error);

                inflight.add_result(&public_key, result);
            });
        }

        rx
    }

    pub fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Pin<Box<dyn Stream<Item = SignedPacket> + Send>> {
        let mut futures = FuturesUnorderedBounded::new(self.relays.len());

        self.relays.iter().for_each(|relay| {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let public_key = public_key.clone();

            futures.push(
                async move { resolve_from_relay(http_client, relay, &public_key, None).await },
            )
        });

        // Box the stream to unify its type
        Box::pin(futures.filter_map(|opt| opt))
    }
}

#[derive(Debug)]
struct InflightpublishRequest {
    signed_packet: SignedPacket,
    senders: Vec<Sender<Result<(), PublishError>>>,
    success_count: usize,
    errors: Vec<PublishError>,
}

#[derive(Clone, Debug)]
pub(crate) struct InflightPublishRequests {
    majority: usize,
    requests: Arc<RwLock<HashMap<PublicKey, InflightpublishRequest>>>,
}

impl InflightPublishRequests {
    fn new(relays_count: usize) -> Self {
        Self {
            majority: (relays_count / 2) + 1,
            requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_sender(
        &self,
        public_key: &PublicKey,
        sender: Sender<Result<(), PublishError>>,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let mut requests = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        if let Some(inflight_request) = requests.get_mut(public_key) {
            if signed_packet.signature() == inflight_request.signed_packet.signature() {
                // Noop, the inflight query is sufficient.
                return Ok(());
            } else if !signed_packet.more_recent_than(&inflight_request.signed_packet) {
                return Err(ConcurrencyError::NotMostRecent)?;
            } else if let Some(cas) = cas {
                if cas != inflight_request.signed_packet.timestamp() {
                    return Err(ConcurrencyError::ConflictRisk)?;
                }
            } else {
                return Err(ConcurrencyError::ConflictRisk)?;
            };

            inflight_request.senders.push(sender.clone());
        } else {
            requests.insert(
                public_key.clone(),
                InflightpublishRequest {
                    signed_packet: signed_packet.clone(),
                    senders: vec![sender],
                    success_count: 0,
                    errors: vec![],
                },
            );
        };

        Ok(())
    }

    pub fn add_result(&self, public_key: &PublicKey, result: Result<(), PublishError>) {
        match result {
            Ok(_) => self.add_success(public_key),
            Err(error) => self.add_error(public_key, error),
        }
    }

    fn add_success(&self, public_key: &PublicKey) {
        let mut inflight = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let mut done = false;

        if let Some(request) = inflight.get_mut(public_key) {
            request.success_count += 1;

            if request.success_count >= self.majority {
                done = true;
            }
        }

        if done {
            if let Some(request) = inflight.remove(public_key) {
                for sender in request.senders {
                    let _ = sender.send(Ok(()));
                }
            }
        }
    }

    fn add_error(&self, public_key: &PublicKey, error: PublishError) {
        let mut inflight = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let mut error_to_send = None;

        if let Some(request) = inflight.get_mut(public_key) {
            request.errors.push(error);

            if request.errors.len() >= self.majority {
                if let Some(most_common_error) = request.errors.first() {
                    if matches!(
                        most_common_error,
                        PublishError::Concurrency(ConcurrencyError::NotMostRecent)
                    ) || matches!(
                        most_common_error,
                        PublishError::Concurrency(ConcurrencyError::CasFailed)
                    ) {
                        error_to_send = Some(most_common_error.clone());
                    }
                };
            }
        }

        if let Some(most_common_error) = error_to_send {
            if let Some(request) = inflight.remove(public_key) {
                for sender in request.senders {
                    let _ = sender.send(Err(most_common_error.clone()));
                }
            }
        }
    }
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
                todo!("an error for both dht error sepcifi and relay bad request")
            }
            StatusCode::CONFLICT => PublishError::Concurrency(ConcurrencyError::NotMostRecent),
            StatusCode::PRECONDITION_FAILED => {
                PublishError::Concurrency(ConcurrencyError::CasFailed)
            }
            StatusCode::PRECONDITION_REQUIRED => {
                PublishError::Concurrency(ConcurrencyError::ConflictRisk)
            }
            StatusCode::INTERNAL_SERVER_ERROR => {
                todo!()
            }
            _ => {
                todo!()
            }
        }
    } else {
        // TODO: better error, a generic fail
        PublishError::Query(QueryError::Timeout)
    }
}
