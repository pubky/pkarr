use std::time::Duration;

use pubky_timestamp::Timestamp;
use reqwest::Client;
use tokio::runtime::{Builder, Runtime};
use url::Url;

use crate::{Cache, CacheKey, PublicKey, SignedPacket};

use crate::client::shared::{publish_to_relay, resolve_from_relay};

use super::PublishError;

pub struct RelaysClient {
    relays: Box<[Url]>,
    http_client: Client,
    cache: Option<Box<dyn Cache>>,
    runtime: Runtime,
}

impl RelaysClient {
    // TODO: allow custom http client?
    pub fn new(relays: Box<[Url]>, cache: Option<Box<dyn Cache>>, timeout: Duration) -> Self {
        Self {
            relays,
            // TODO: allow passing a runtime.
            runtime: Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime"),
            http_client: Client::builder()
                .timeout(timeout)
                .build()
                .expect("Client building should be infallible"),
            cache,
        }
    }

    pub fn publish(
        &self,
        signed_packet: &SignedPacket,
        sender: flume::Sender<Result<(), PublishError>>,
        cas: Option<Timestamp>,
    ) {
        let public_key = signed_packet.public_key();
        let body = signed_packet.to_relay_payload();
        let cas = cas.map(|timestamp| timestamp.format_http_date());

        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let public_key = public_key.clone();
            let body = body.clone();
            let sender = sender.clone();
            let cas = cas.clone();

            self.runtime.spawn(async move {
                if publish_to_relay(http_client, relay, &public_key, body, cas)
                    .await
                    .is_ok()
                {
                    let _ = sender.send(Ok(()));
                }
            });
        }
    }

    pub fn resolve(
        &self,
        public_key: &PublicKey,
        cache_key: &CacheKey,
        sender: flume::Sender<SignedPacket>,
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

#[cfg(test)]
mod tests {
    //! Relays only tests

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
            .dht_config(mainline::Config {
                request_timeout: Duration::from_millis(10),
                ..Default::default()
            })
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

    // #[tokio::test]
    async fn concurrent_publish_different() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .no_default_network()
            .relays(Some(vec![relay.local_url()]))
            .request_timeout(Duration::from_millis(100))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let clone = client.clone();

        let handle = tokio::spawn(async move {
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
}
