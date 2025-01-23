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

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or
/// [Relays](https://pkarr.org/relays).
pub struct Client {
    sender: Sender<ActorMessage>,
    cache: Option<Box<dyn Cache>>,
}

impl Client {
    pub fn new(config: Config) -> Result<Client, BuildError> {
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
            .spawn(move || actor_thread(receiver, cache_clone, config))
            .map_err(BuildError::ActorThreadSpawn)?;

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

    // === Publish ===

    /// Publishes a [SignedPacket] to the [mainline] Dht and or [Relays](https://pkarr.org/relays).
    ///
    /// # Consistency
    ///
    /// Publishing different packets concurrently is not safe, as it risks losing data (records)
    /// in the earlier packet that are accidentally gets dropped in the more recently published
    /// packet.
    ///
    /// We can't perfectly protect against this, as the distributed network is available and
    /// partition tolerant, and thus isn't consistent. See [CAP theorem](https://en.wikipedia.org/wiki/CAP_theorem).
    ///
    /// However, there are ways to reduce the risk of inconsistent writes:
    /// - You can and should "read before write" by calling [Self::resolve_most_recent]
    ///     before publishing, to reduce the chances of missing records from previous packets
    ///     published from other clients.
    /// - Publishing two different [SignedPacket]s from the same client concurrently will return
    ///    an [PublishError::ConcurrentPublish] error.
    /// - Before publishing, (if you don't manually call [Self::resolve_most_recent]) this method
    ///     will query the network first, and if it finds a more recent [SignedPacket] than the packet
    ///     you are trying to publish, it will return a [PublishError::NotMostRecent] error.
    /// - Before publishing, (if you don't manually call [Self::resolve_most_recent]) this method
    ///     will query the network first, and if it finds a more recent [SignedPacket] than what
    ///     already existed in cache before publishing, it will return a [PublishError::CasFailed] error.
    /// - While Publishing, if most nodes or relays responds with a more recent [SignedPacket]s,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - While Publishing, if most nodes and/or relays responds with a compare and swap error,
    ///     which may happen if they know of a more recent [SignedPacket] than the one we
    ///     discovered before publishing, this method will return a [PublishError::NotMostRecent] error.
    ///     This error is not as reliable as it only works if all nodes or all relays agree, to
    ///     avoid disruption from malicious nodes/relays.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        let cache_key = CacheKey::from(signed_packet.public_key());

        let cached = self.check_most_recent(&cache_key, signed_packet)?;

        // TODO: skip lookup if the cache is still very fresh, persumably the user
        // called [Self::resolve_most_recent] very recently.
        // if cahed.map(|s|s.last_seen() )
        let _ = self
            .resolve_most_recent(&signed_packet.public_key())
            .await?;

        self.publish_inner(signed_packet, cache_key, cached.map(|s| s.timestamp()))?
            .recv_async()
            .await
            .expect("Query was dropped before sending a response, please open an issue.")
    }

    /// Publishes a [SignedPacket] to the [mainline] Dht and or [Relays](https://pkarr.org/relays).
    ///
    /// # Consistency
    ///
    /// Publishing different packets concurrently is not safe, as it risks losing data (records)
    /// in the earlier packet that are accidentally gets dropped in the more recently published
    /// packet.
    ///
    /// We can't perfectly protect against this, as the distributed network is available and
    /// partition tolerant, and thus isn't consistent. See [CAP theorem](https://en.wikipedia.org/wiki/CAP_theorem).
    ///
    /// However, there are ways to reduce the risk of inconsistent writes:
    /// - You can and should "read before write" by calling [Self::resolve_most_recent_sync]
    ///     before publishing, to reduce the chances of missing records from previous packets
    ///     published from other clients.
    /// - Publishing two different [SignedPacket]s from the same client concurrently will return
    ///    an [PublishError::ConcurrentPublish] error.
    /// - Before publishing, (if you don't manually call [Self::resolve_most_recent_sync]) this method
    ///     will query the network first, and if it finds a more recent [SignedPacket] than the packet
    ///     you are trying to publish, it will return a [PublishError::NotMostRecent] error.
    /// - Before publishing, (if you don't manually call [Self::resolve_most_recent_sync]) this method
    ///     will query the network first, and if it finds a more recent [SignedPacket] than what
    ///     already existed in cache before publishing, it will return a [PublishError::CasFailed] error.
    /// - While Publishing, if most nodes or relays responds with a more recent [SignedPacket]s,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - While Publishing, if most nodes and/or relays responds with a compare and swap error,
    ///     which may happen if they know of a more recent [SignedPacket] than the one we
    ///     discovered before publishing, this method will return a [PublishError::NotMostRecent] error.
    ///     This error is not as reliable as it only works if all nodes or all relays agree, to
    ///     avoid disruption from malicious nodes/relays.
    pub fn publish_sync(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        let cache_key = CacheKey::from(signed_packet.public_key());

        let cached = self.check_most_recent(&cache_key, signed_packet)?;

        let _ = self.resolve_most_recent_sync(&signed_packet.public_key())?;

        self.publish_inner(signed_packet, cache_key, cached.map(|s| s.timestamp()))?
            .recv()
            .expect("Query was dropped before sending a response, please open an issue.")
    }

    // === Resolve ===

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

    /// Returns a [Iterator] of incoming [SignedPacket]s.
    pub fn resolve_iter(
        &self,
        public_key: &PublicKey,
    ) -> Result<SignedPacketIterator, ClientWasShutdown> {
        Ok(SignedPacketIterator(self.resolve_rx(public_key)?))
    }

    /// Returns a [Stream] of incoming [SignedPacket]s.
    pub fn resolve_stream(
        &self,
        public_key: &PublicKey,
    ) -> Result<SignedPacketStream, ClientWasShutdown> {
        Ok(self.resolve_rx(public_key)?.into())
    }

    /// Returns the most recent [SignedPacket] found after querying all
    /// [mainline] Dht nodes and or [Relays](https:://pkarr.org/relays).
    ///
    /// Useful if you want to read the most recent packet before publishing
    /// a new packet.
    ///
    /// This is a best effort, and doesn't guarantee consistency.
    pub async fn resolve_most_recent(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        let mut stream = self.resolve_stream(public_key)?;
        while (stream.next().await).is_some() {}

        Ok(self.cache().and_then(|cache| cache.get(&public_key.into())))
    }

    /// Returns the most recent [SignedPacket] found after querying all
    /// [mainline] Dht nodes and or [Relays](https:://pkarr.org/relays).
    ///
    /// Useful if you want to read the most recent packet before publishing
    /// a new packet.
    ///
    /// This is a best effort, and doesn't guarantee consistency.
    pub fn resolve_most_recent_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        let mut iter = self.resolve_iter(public_key)?;
        while (iter.next()).is_some() {}

        Ok(self.cache().and_then(|cache| cache.get(&public_key.into())))
    }

    // === Shutdwon ===

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
            .send(ActorMessage::Publish(signed_packet.clone(), sender, cas))
            .map_err(|_| PublishError::ClientWasShutdown)?;

        Ok(receiver)
    }

    fn check_most_recent(
        &self,
        cache_key: &CacheKey,
        signed_packet: &SignedPacket,
    ) -> Result<Option<SignedPacket>, PublishError> {
        if let Some(cached) = self.cache.as_ref().and_then(|cache| cache.get(cache_key)) {
            if cached.timestamp() >= signed_packet.timestamp() {
                return Err(PublishError::NotMostRecent);
            }

            return Ok(Some(cached));
        }

        Ok(None)
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

#[derive(thiserror::Error, Debug)]
/// Errors occuring during building a [Client]
pub enum BuildError {
    #[error("Failed to spawn the actor thread.")]
    /// Failed to spawn the actor thread.
    ActorThreadSpawn(std::io::Error),

    #[error("Failed to bind mainline UdpSocket (and Relays are disabled).")]
    /// Failed to bind mainline UdpSocket (and Relays are disabled).
    MainlineUdpSocket(std::io::Error),

    #[error("Client configured without Mainline node or relays.")]
    /// Client configured without Mainline node or relays.
    NoNetwork,
}

#[derive(Debug)]
pub struct ClientWasShutdown;

impl std::error::Error for ClientWasShutdown {}

impl std::fmt::Display for ClientWasShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pkarr Client was shutdown")
    }
}

#[derive(thiserror::Error, Debug, Clone)]
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

    #[error("Publish query is already inflight for the same public_key with a different value")]
    /// [crate::Client::publish] is already inflight to the same public_key with a different value
    ConcurrentPublish,

    // === Mainline only errors ===
    //
    #[cfg(feature = "dht")]
    #[error("Publishing SignedPacket to Mainline failed.")]
    ///Publishing SignedPacket to Mainline failed.
    NoClosestNodes,
}

impl From<ClientWasShutdown> for PublishError {
    fn from(_: ClientWasShutdown) -> Self {
        Self::ClientWasShutdown
    }
}

#[cfg(test)]
mod tests {
    //! Combined client tests

    use std::{thread, time::Duration};

    use native::{BuildError, PublishError};
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
    async fn concurrent_publish_different() {
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

    #[tokio::test]
    async fn no_network() {
        assert!(matches!(
            Client::builder().no_default_network().build(),
            Err(BuildError::NoNetwork)
        ));
    }
}
