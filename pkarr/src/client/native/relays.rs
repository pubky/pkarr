use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use ed25519_dalek::Signature;
use flume::Sender;
use pubky_timestamp::Timestamp;
use reqwest::{Client, StatusCode};
use tokio::runtime::{Builder, Runtime};
use url::Url;

use crate::{Cache, CacheKey, PublicKey, SignedPacket};

use crate::client::shared::{publish_to_relay, resolve_from_relay};

use super::PublishError;

pub struct RelaysClient {
    relays: Box<[Url]>,
    http_client: Client,
    cache: Option<Box<dyn Cache>>,
    runtime: Arc<Runtime>,

    inflight_publish: InflightPublishRequests,
}

impl RelaysClient {
    pub fn new(
        relays: Box<[Url]>,
        cache: Option<Box<dyn Cache>>,
        timeout: Duration,
        runtime: Option<Arc<Runtime>>,
    ) -> Self {
        let inflight_publish = InflightPublishRequests::new(relays.len());

        Self {
            relays,
            runtime: runtime.unwrap_or(Arc::new(
                Builder::new_multi_thread()
                    .worker_threads(4)
                    .enable_all()
                    .build()
                    .expect("Failed to create Tokio runtime"),
            )),
            http_client: Client::builder()
                .timeout(timeout)
                .build()
                .expect("Client building should be infallible"),
            cache,

            inflight_publish,
        }
    }

    pub fn publish(
        &self,
        signed_packet: &SignedPacket,
        sender: Sender<Result<(), PublishError>>,
        cas: Option<Timestamp>,
    ) {
        let public_key = signed_packet.public_key();
        let body = signed_packet.to_relay_payload();
        let cas = cas.map(|timestamp| timestamp.format_http_date());

        if let Err(error) =
            self.inflight_publish
                .add_sender(&public_key, signed_packet.signature(), &sender)
        {
            // ConcurrentPublish
            let _ = sender.send(Err(error));
            return;
        };

        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let public_key = public_key.clone();
            let body = body.clone();
            let cas = cas.clone();
            let inflight = self.inflight_publish.clone();

            self.runtime.spawn(async move {
                inflight.add_result(
                    &public_key,
                    publish_to_relay(http_client, relay, &public_key, body, cas).await,
                )
            });
        }
    }

    pub fn resolve(
        &self,
        public_key: &PublicKey,
        cache_key: &CacheKey,
        sender: Sender<SignedPacket>,
    ) {
        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let sender = sender.clone();
            let cache = self.cache.clone();
            let cache_key = *cache_key;
            let public_key = public_key.clone();

            self.runtime.spawn(async move {
                if let Ok(Some(signed_packet)) =
                    resolve_from_relay(http_client, relay, &public_key, cache, &cache_key).await
                {
                    let _ = sender.send(signed_packet);
                }
            });
        }
    }
}

struct InflightpublishRequest {
    signature: Signature,
    senders: Vec<Sender<Result<(), PublishError>>>,
    success_count: usize,
    errors: Vec<PublishError>,
}

#[derive(Clone)]
struct InflightPublishRequests {
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

    pub fn contains_key(&self, public_key: &PublicKey) -> bool {
        self.requests
            .read()
            .expect("InflightPublishRequests read lock")
            .contains_key(public_key)
    }

    pub fn add_sender(
        &self,
        public_key: &PublicKey,
        signature: Signature,
        sender: &Sender<Result<(), PublishError>>,
    ) -> Result<(), PublishError> {
        let mut requests = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let inflight_request =
            requests
                .entry(public_key.clone())
                .or_insert(InflightpublishRequest {
                    signature,
                    senders: vec![],
                    success_count: 0,
                    errors: vec![],
                });

        if inflight_request.signature != signature {
            return Err(PublishError::ConcurrentPublish);
        }

        inflight_request.senders.push(sender.clone());

        Ok(())
    }

    pub fn add_result(&self, public_key: &PublicKey, result: Result<(), reqwest::Error>) {
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

    fn add_error(&self, public_key: &PublicKey, error: reqwest::Error) {
        let mut inflight = self
            .requests
            .write()
            .expect("InflightPublishRequests write lock");

        let mut error_to_send = None;

        if let Some(request) = inflight.get_mut(public_key) {
            request.errors.push(if error.is_timeout() {
                PublishError::Timeout
            } else if error.is_status() {
                match error
                    .status()
                    .expect("previously verified that it is a status error")
                {
                    StatusCode::CONFLICT => PublishError::NotMostRecent,
                    StatusCode::PRECONDITION_FAILED => PublishError::CasFailed,
                    StatusCode::TOO_MANY_REQUESTS => {
                        todo!()
                    }
                    StatusCode::INTERNAL_SERVER_ERROR => {
                        todo!()
                    }
                    _ => {
                        todo!()
                    }
                }
            } else {
                todo!()
            });

            if request.errors.len() >= self.majority {
                if let Some(most_common_error) = request.errors.first() {
                    if matches!(most_common_error, PublishError::NotMostRecent)
                        || matches!(most_common_error, PublishError::CasFailed)
                    {
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

#[cfg(test)]
mod tests {
    //! Relays only tests

    use std::sync::Arc;
    use std::time::Duration;

    use pkarr_relay::Relay;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    #[tokio::test]
    async fn publish_resolve() {
        let relay = Relay::start_test().await.unwrap();

        let a = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[tokio::test]
    async fn thread_safe() {
        let relay = Relay::start_test().await.unwrap();

        let a = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        tokio::spawn(async move {
            let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
            assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

            let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
            assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
            assert_eq!(from_cache.last_seen(), resolved.last_seen());
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn return_expired_packet_fallback() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .request_timeout(Duration::from_millis(10))
            // Everything is expired
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client
            .cache()
            .unwrap()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }

    #[tokio::test]
    async fn ttl_0_test() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client.publish(&signed_packet).await.unwrap();

        // First Call
        let resolved = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

        thread::sleep(Duration::from_millis(10));

        let second = client
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
    }

    #[tokio::test]
    async fn not_found() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .request_timeout(Duration::from_millis(20))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, None);
    }
    #[tokio::test]
    async fn empty_list_of_relays() {
        let result = Client::builder()
            .no_default_network()
            .relays(Some(vec![]))
            .build();

        assert!(matches!(result, Err(BuildError::EmptyListOfRelays)))
    }

    #[test]
    async fn concurrent_publish_different() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let clone = client.clone();

        // Avoid calling resolve_most_recent again inside publish
        let _ = clone.resolve_most_recent(&signed_packet.public_key()).await;

        let handle = rt.spawn(async move {
            let signed_packet = SignedPacket::builder()
                .txt("foo".try_into().unwrap(), "zar".try_into().unwrap(), 30)
                .sign(&keypair)
                .unwrap();

            let result = clone.publish(&signed_packet).await;

            assert!(matches!(result, Err(PublishError::ConcurrentPublish)));
        });

        client.publish(&signed_packet).await.unwrap();
        handle.await.unwrap()
    }

    #[tokio::test]
    async fn not_most_recent() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let older = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let more_recent = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        client.publish(&more_recent).await.unwrap();

        let result = client.publish(&older).await;

        assert!(matches!(result, Err(PublishError::NotMostRecent)));
    }

    #[tokio::test]
    async fn cas_failed() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let older = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let cas = Timestamp::now();

        let more_recent = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        client.publish(&older).await.unwrap();

        let result = client.publish_with_cas(&more_recent, Some(cas)).await;

        assert!(matches!(result, Err(PublishError::CasFailed)));
    }
}
