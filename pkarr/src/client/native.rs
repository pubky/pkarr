//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or [Relays](https://pkarr.org/relays).

use flume::r#async::RecvStream;
use flume::{Receiver, Sender};
use futures_lite::{Stream, StreamExt};
use pubky_timestamp::Timestamp;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::{num::NonZeroUsize, thread};
use tracing::debug;

use crate::{Cache, CacheKey, InMemoryCache};
use crate::{PublicKey, SignedPacket};

mod actor_thread;
mod builder;
#[cfg(feature = "dht")]
mod dht;
#[cfg(feature = "relays")]
mod relays;

pub use actor_thread::Info;
use actor_thread::{actor_thread, ActorMessage};
pub use builder::{ClientBuilder, Config};

// TODO: recognize when the cache is updated since `publish` was called,
// and return an error ... a CAS error maybe.

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or
/// [Relays](https://pkarr.org/relays).
pub struct Client {
    sender: Sender<ActorMessage>,
    cache: Option<Box<dyn Cache>>,
}

impl Client {
    pub fn new(config: Config) -> Result<Client, std::io::Error> {
        let (sender, receiver) = flume::bounded(32);

        let cache = if config.cache_size == 0 {
            None
        } else {
            Some(
                config.cache.clone().unwrap_or(Box::new(InMemoryCache::new(
                    NonZeroUsize::new(config.cache_size)
                        .expect("if cache size is zero cache should be disabled."),
                ))),
            )
        };

        let cache_clone = cache.clone();

        let client = Client { sender, cache };

        debug!(?config, "Starting Client main loop..");

        thread::Builder::new()
            .name("Pkarr Dht actor thread".to_string())
            .spawn(move || actor_thread(receiver, cache_clone, config))?;

        let (tx, rx) = flume::bounded(1);

        client
            .sender
            .send(ActorMessage::Check(tx))
            .expect("actor thread unexpectedly shutdown");

        rx.recv().expect("infallible")?;

        Ok(client)
    }

    /// Returns a builder to edit config before creating Client.
    ///
    /// You can use [ClientBuilder::no_default_network] to start from a clean slate and
    /// decide which networks to use.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    // === Getters ===

    /// Returns [Info] about the running session from the actor thread.
    pub fn info(&self) -> Result<Info, ClientWasShutdown> {
        let (tx, rx) = flume::bounded(1);

        self.sender
            .send(ActorMessage::Info(tx))
            .map_err(|_| ClientWasShutdown)?;

        rx.recv().map_err(|_| ClientWasShutdown)
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> Option<&dyn Cache> {
        self.cache.as_deref()
    }

    // === Public Methods ===

    // TODO: Test failed to publish
    /// Publishes a [SignedPacket] to the Dht.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        let cache_key = CacheKey::from(signed_packet.public_key());

        let cas = self.check_most_recent(&cache_key, signed_packet)?;

        let mut stream = self.resolve_stream(&signed_packet.public_key())?;
        while (stream.next().await).is_some() {}

        self.publish_inner(signed_packet, cache_key, cas)?
            .recv_async()
            .await
            .expect("Query was dropped before sending a response, please open an issue.")
    }

    /// Publishes a [SignedPacket] to the Dht.
    pub fn publish_sync(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        let cache_key = CacheKey::from(signed_packet.public_key());

        let cas = self.check_most_recent(&cache_key, signed_packet)?;

        let mut iter = self.resolve_iter(&signed_packet.public_key())?;
        while (iter.next()).is_some() {}

        self.publish_inner(signed_packet, cache_key, cas)?
            .recv()
            .expect("Query was dropped before sending a response, please open an issue.")
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_iter], or [Self::resolve_stream]
    /// to iterate over incoming [SignedPacket]s until your lookup criteria is satisfied.
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv_async().await.ok())
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to have more control, you can call [Self::resolve_iter], or [Self::resolve_stream]
    /// to iterate over incoming [SignedPacket]s until your lookup criteria is satisfied.
    pub fn resolve_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.recv().ok())
    }

    pub fn resolve_iter(
        &self,
        public_key: &PublicKey,
    ) -> Result<SignedPacketIterator, ClientWasShutdown> {
        Ok(SignedPacketIterator(self.resolve_rx(public_key)?))
    }

    pub fn resolve_stream(
        &self,
        public_key: &PublicKey,
    ) -> Result<SignedPacketStream, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.into())
    }

    /// Shutdown the actor thread loop.
    pub async fn shutdown(&mut self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv_async().await;
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown_sync(&self) {
        let (sender, receiver) = flume::bounded(1);

        let _ = self.sender.send(ActorMessage::Shutdown(sender));
        let _ = receiver.recv();
    }

    // === Private Methods ===

    /// Returns a `flume::Receiver<SignedPacket>` that allows [iterating](flume::Receiver::recv) over or
    /// [streaming](flume::Receiver::recv_async) incoming [SignedPacket]s, in case you need more control over your
    /// caching strategy and when resolution should terminate, as well as filtering [SignedPacket]s according to a custom criteria.
    pub(crate) fn resolve_rx(
        &self,
        public_key: &PublicKey,
    ) -> Result<Receiver<SignedPacket>, ClientWasShutdown> {
        let (tx, rx) = flume::bounded::<SignedPacket>(1);

        self.sender
            .send(ActorMessage::Resolve(public_key.clone(), tx.clone()))
            .map_err(|_| ClientWasShutdown)?;

        Ok(rx)
    }

    pub(crate) fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
        cache_key: CacheKey,
        cas: Option<Timestamp>,
    ) -> Result<Receiver<Result<(), PublishError>>, PublishError> {
        if let Some(cas) = cas {
            if let Some(cached) = self.cache.as_ref().and_then(|cache| cache.get(&cache_key)) {
                if cached.timestamp() != cas {
                    return Err(PublishError::CasFailed);
                }
            }
        }

        let (sender, receiver) = flume::bounded::<Result<(), PublishError>>(1);

        self.sender
            .send(ActorMessage::Publish(signed_packet.clone(), sender))
            .map_err(|_| PublishError::ClientWasShutdown)?;

        Ok(receiver)
    }

    fn check_most_recent(
        &self,
        cache_key: &CacheKey,
        signed_packet: &SignedPacket,
    ) -> Result<Option<Timestamp>, PublishError> {
        if let Some(cached) = self.cache.as_ref().and_then(|cache| cache.get(cache_key)) {
            if cached.timestamp() >= signed_packet.timestamp() {
                return Err(PublishError::NotMostRecent);
            }

            return Ok(Some(cached.timestamp()));
        }

        Ok(None)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ClientWasShutdown;

impl std::error::Error for ClientWasShutdown {}

impl std::fmt::Display for ClientWasShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pkarr Client was shutdown")
    }
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
/// Errors occuring during publishing a [SignedPacket]
pub enum PublishError {
    #[error("Found a more recent SignedPacket in the client's cache")]
    /// Found a more recent SignedPacket in the client's cache
    NotMostRecent,

    #[error("Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing")]
    /// Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing
    CasFailed,

    #[error("Pkarr Client was shutdown")]
    ClientWasShutdown,

    // TODO: should we remove this when there is no dht or support it for relays?
    #[error("Publish query is already inflight for the same public_key")]
    /// [crate::Client::publish] is already inflight to the same public_key
    PublishInflight,
}

impl From<ClientWasShutdown> for PublishError {
    fn from(_: ClientWasShutdown) -> Self {
        Self::ClientWasShutdown
    }
}

pub struct SignedPacketIterator(flume::Receiver<SignedPacket>);

impl Iterator for SignedPacketIterator {
    type Item = SignedPacket;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.recv().ok()
    }
}

pub struct SignedPacketStream(RecvStream<'static, SignedPacket>);

impl From<Receiver<SignedPacket>> for SignedPacketStream {
    fn from(value: Receiver<SignedPacket>) -> Self {
        Self(value.into_stream())
    }
}

impl Stream for SignedPacketStream {
    type Item = SignedPacket;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    //! Combined client tests

    use std::{thread, time::Duration};

    use native::PublishError;
    use pkarr_relay::Relay;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    #[tokio::test]
    async fn publish_resolve() {
        let relay = Relay::start_test().await.unwrap();

        let a = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
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
            .bootstrap(&[relay.resolver_address().to_string()])
            .relays(Some(vec![relay.local_url()]))
            .request_timeout(Duration::from_millis(20))
            .build()
            .unwrap();

        let keypair = Keypair::random();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn inflight() {
        let relay = Relay::start_test().await.unwrap();

        let client = Client::builder()
            .bootstrap(&[relay.resolver_address().to_string()])
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
        let packet = signed_packet.clone();

        let handle = tokio::spawn(async move {
            let result = clone.publish(&packet).await;

            assert_eq!(result, Err(PublishError::PublishInflight));
        });

        client.publish(&signed_packet).await.unwrap();

        handle.await.unwrap()
    }
}
