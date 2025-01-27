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

        #[cfg(feature = "relays")]
        if config
            .relays
            .as_ref()
            .map(|relays| relays.is_empty())
            .unwrap_or_default()
        {
            return Err(BuildError::EmptyListOfRelays);
        }

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

    /// Turn this node's routing table to a list of bootstraping nodes.
    ///
    /// Returns `Ok(None)` if the only relays are used.
    pub fn to_bootstrap(&self) -> Result<Option<Vec<String>>, ClientWasShutdown> {
        let (sender, receiver) = flume::bounded::<Option<Vec<String>>>(1);

        self.sender
            .send(ActorMessage::ToBootstrap(sender))
            .map_err(|_| ClientWasShutdown)?;

        receiver.recv().map_err(|_| ClientWasShutdown)
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
    /// - If the cached [SignedPacket] is more recent than the packet you are trying to publish,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - Before publishing this method will refresh the cache by resolving the most recent [SignedPacket]
    ///     if needed (if you didn't call [Self::resolve_most_recent] right before publishing),
    ///     and if it finds a more recent [SignedPacket] than the cached packet before before
    ///     resolving the most recent packet, it will return a [PublishError::CasFailed] error.
    /// - Publishing two different [SignedPacket]s from the same client concurrently will return
    ///    a [PublishError::ConcurrentPublish] error.
    /// - If all pre-checks above passed, and publishing starts, if most dht nodes or relays responded
    ///     claims to have more recent packet than the one we are trying to publish,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - If most dht nodes or relays responds with a compare and swap error,
    ///     which may happen if they know of a more recent [SignedPacket] than the one we read from
    ///     our fresh cache before publishing, this method will return a [PublishError::NotMostRecent] error.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(
            signed_packet,
            self.resolve_most_recent(&signed_packet.public_key())
                .await?
                .map(|s| s.timestamp()),
        )?
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
    /// - If the cached [SignedPacket] is more recent than the packet you are trying to publish,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - Before publishing this method will refresh the cache by resolving the most recent [SignedPacket]
    ///     if needed (if you didn't call [Self::resolve_most_recent_sync] right before publishing),
    ///     and if it finds a more recent [SignedPacket] than the cached packet before before
    ///     resolving the most recent packet, it will return a [PublishError::CasFailed] error.
    /// - Publishing two different [SignedPacket]s from the same client concurrently will return
    ///    a [PublishError::ConcurrentPublish] error.
    /// - If all pre-checks above passed, and publishing starts, if most dht nodes or relays responded
    ///     claims to have more recent packet than the one we are trying to publish,
    ///     this method will return a [PublishError::NotMostRecent] error.
    /// - If most dht nodes or relays responds with a compare and swap error,
    ///     which may happen if they know of a more recent [SignedPacket] than the one we read from
    ///     our fresh cache before publishing, this method will return a [PublishError::NotMostRecent] error.
    pub fn publish_sync(&self, signed_packet: &SignedPacket) -> Result<(), PublishError> {
        self.publish_inner(
            signed_packet,
            self.resolve_most_recent_sync(&signed_packet.public_key())?
                .map(|s| s.timestamp()),
        )?
        .recv()
        .expect("Query was dropped before sending a response, please open an issue.")
    }

    /// Publish a [SignedPacket] with a manually provided CAS timestamp.
    ///
    /// Useful for:
    /// 1. Manually call [Self::resolve_most_recent] and use the result as `CAS` field
    ///     instead of relying on [Self::publish] assuming that you are publishing based
    ///     on the cached signed packet which could be updated immediately after you call
    ///     [Self::publish] defeating the purpose of `CAS`.
    /// 2. Relays that get a request to publish a packet from a remote client that
    ///     sends their CAS as `IF_UNMODIFIED_SINCE` header.
    pub async fn publish_with_cas(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        self.publish_inner(signed_packet, cas)?
            .recv_async()
            .await
            .expect("Query was dropped before sending a response, please open an issue.")
    }

    /// Publish a [SignedPacket] with a manually provided CAS timestamp.
    ///
    /// Useful for relays that get a request to publish a packet from a remote client that
    /// sends their CAS as `IF_UNMODIFIED_SINCE` header.
    pub fn publish_with_cas_sync(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        self.publish_inner(signed_packet, cas)?
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

    fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<Receiver<Result<(), PublishError>>, PublishError> {
        let cache_key: CacheKey = signed_packet.public_key().into();

        self.check_conflict(signed_packet, &cache_key, cas)?;

        if let Some(cache) = self.cache() {
            cache.put(&cache_key, signed_packet);
        }

        let (sender, receiver) = flume::bounded::<Result<(), PublishError>>(1);

        self.sender
            .send(ActorMessage::Publish(signed_packet.clone(), sender, cas))
            .map_err(|_| PublishError::ClientWasShutdown)?;

        Ok(receiver)
    }

    fn check_conflict(
        &self,
        signed_packet: &SignedPacket,
        cache_key: &CacheKey,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        if let Some(cached) = self.cache.as_ref().and_then(|cache| cache.get(cache_key)) {
            if cached.timestamp() >= signed_packet.timestamp() {
                return Err(ConcurrencyError::NotMostRecent)?;
            }
        } else if let Some(cas) = cas {
            if let Some(cached) = self.cache.as_ref().and_then(|cache| cache.get(cache_key)) {
                if cached.timestamp() != cas {
                    return Err(ConcurrencyError::CasFailed)?;
                }
            }
        }

        Ok(())
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

    #[error("Client configured without Mainline node or relays.")]
    /// Client configured without Mainline node or relays.
    NoNetwork,

    #[cfg(feature = "dht")]
    #[error("Failed to bind mainline UdpSocket (and Relays are disabled).")]
    /// Failed to bind mainline UdpSocket (and Relays are disabled).
    MainlineUdpSocket(std::io::Error),

    #[cfg(feature = "relays")]
    #[error("Passed an empty list of relays")]
    /// Passed an empty list of relays
    EmptyListOfRelays,
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
    #[error("Pkarr Client was shutdown")]
    ClientWasShutdown,

    #[error(transparent)]
    Concurrency(#[from] ConcurrencyError),

    /// Publish query timed out with no responses neither success or errors, from Dht or relays.
    #[error("Publish query timed out with no responses neither success or errors.")]
    Timeout,

    // === Mainline only errors ===
    //
    #[cfg(feature = "dht")]
    #[error("Publishing SignedPacket to Mainline failed.")]
    ///Publishing SignedPacket to Mainline failed.
    NoClosestNodes,

    #[cfg(feature = "dht")]
    #[error("Publishing SignedPacket to Mainline failed.")]
    ///Publishing SignedPacket to Mainline failed, received an error response.
    MainlineErrorResponse(mainline::rpc::messages::ErrorSpecific),
}

#[derive(thiserror::Error, Debug, Clone)]
/// Errors that requires resolving most recent [SignedPacket] before publishing.
pub enum ConcurrencyError {
    #[error("A different SignedPacket is being concurrently published for the same PublicKey.")]
    /// A different [SignedPacket] is being concurrently published for the same [PublicKey].
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket] before publishing again.
    ConflictRisk,

    #[error("Found a more recent SignedPacket in the client's cache")]
    /// Found a more recent SignedPacket in the client's cache
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket] before publishing again.
    NotMostRecent,

    #[error("Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing")]
    /// Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket] before publishing again.
    CasFailed,
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

    use native::{ConcurrencyError, PublishError};
    use pkarr_relay::Relay;
    use rstest::rstest;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    enum Networks {
        Dht,
        Relays,
        Both,
    }

    /// Parametric [ClientBuilder] with no default networks,
    /// instead it uses mainline or relays depending on `networks` enum.
    fn builder(relay: &Relay, networks: &Networks) -> ClientBuilder {
        let builder = Client::builder()
            .no_default_network()
            .request_timeout(Duration::from_millis(100));

        match networks {
            Networks::Dht => builder
                .bootstrap(relay.as_bootstrap())
                .resolvers(Some(vec![relay.resolver_address().to_string()])),
            Networks::Relays => builder.relays(Some(vec![relay.local_url()])),
            Networks::Both => builder
                .bootstrap(&[relay.resolver_address().to_string()])
                .resolvers(Some(vec![relay.resolver_address().to_string()]))
                .relays(Some(vec![relay.local_url()])),
        }
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    // #[case::relays(Networks::Relays)]
    #[case::both_networks(Networks::Both)]
    #[tokio::test]
    async fn publish_resolve(#[case] networks: Networks) {
        let relay = Relay::start_test().await.unwrap();

        let a = builder(&relay, &networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = builder(&relay, &networks).build().unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    // #[tokio::test]
    // async fn thread_safe() {
    //     let relay = Relay::start_test().await.unwrap();
    //
    //     let a = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .build()
    //         .unwrap();
    //
    //     let keypair = Keypair::random();
    //
    //     let signed_packet = SignedPacket::builder()
    //         .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
    //         .sign(&keypair)
    //         .unwrap();
    //
    //     a.publish(&signed_packet).await.unwrap();
    //
    //     let b = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .build()
    //         .unwrap();
    //
    //     tokio::spawn(async move {
    //         let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
    //         assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    //
    //         let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
    //         assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
    //         assert_eq!(from_cache.last_seen(), resolved.last_seen());
    //     })
    //     .await
    //     .unwrap();
    // }
    //
    // #[tokio::test]
    // async fn return_expired_packet_fallback() {
    //     let relay = Relay::start_test().await.unwrap();
    //
    //     let client = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .dht_config(mainline::Config {
    //             request_timeout: Duration::from_millis(10),
    //             ..Default::default()
    //         })
    //         // Everything is expired
    //         .maximum_ttl(0)
    //         .build()
    //         .unwrap();
    //
    //     let keypair = Keypair::random();
    //
    //     let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
    //
    //     client
    //         .cache()
    //         .unwrap()
    //         .put(&keypair.public_key().into(), &signed_packet);
    //
    //     let resolved = client.resolve(&keypair.public_key()).await.unwrap();
    //
    //     assert_eq!(resolved, Some(signed_packet));
    // }
    //
    // #[tokio::test]
    // async fn ttl_0_test() {
    //     let relay = Relay::start_test().await.unwrap();
    //
    //     let client = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .maximum_ttl(0)
    //         .build()
    //         .unwrap();
    //
    //     let keypair = Keypair::random();
    //     let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
    //
    //     client.publish(&signed_packet).await.unwrap();
    //
    //     // First Call
    //     let resolved = client
    //         .resolve(&signed_packet.public_key())
    //         .await
    //         .unwrap()
    //         .unwrap();
    //
    //     assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());
    //
    //     thread::sleep(Duration::from_millis(10));
    //
    //     let second = client
    //         .resolve(&signed_packet.public_key())
    //         .await
    //         .unwrap()
    //         .unwrap();
    //     assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
    // }
    //
    // #[tokio::test]
    // async fn not_found() {
    //     let relay = Relay::start_test().await.unwrap();
    //
    //     let client = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .request_timeout(Duration::from_millis(20))
    //         .build()
    //         .unwrap();
    //
    //     let keypair = Keypair::random();
    //
    //     let resolved = client.resolve(&keypair.public_key()).await.unwrap();
    //
    //     assert_eq!(resolved, None);
    // }
    //
    // #[tokio::test]
    // async fn no_network() {
    //     assert!(matches!(
    //         Client::builder().no_default_network().build(),
    //         Err(BuildError::NoNetwork)
    //     ));
    // }
    //
    // #[tokio::test]
    // async fn concurrent_publish_different() {
    //     let relay = Relay::start_test().await.unwrap();
    //
    //     let client = Client::builder()
    //         .bootstrap(&[relay.resolver_address().to_string()])
    //         .relays(Some(vec![relay.local_url()]))
    //         .request_timeout(Duration::from_millis(100))
    //         .build()
    //         .unwrap();
    //
    //     let keypair = Keypair::random();
    //
    //     let signed_packet = SignedPacket::builder()
    //         .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
    //         .sign(&keypair)
    //         .unwrap();
    //
    //     let clone = client.clone();
    //
    //     let handle = tokio::spawn(async move {
    //         let signed_packet = SignedPacket::builder()
    //             .txt("foo".try_into().unwrap(), "zar".try_into().unwrap(), 30)
    //             .sign(&keypair)
    //             .unwrap();
    //
    //         let result = clone.publish(&signed_packet).await;
    //
    //         assert!(matches!(
    //             result,
    //             Err(PublishError::Concurrency(ConcurrencyError::ConflictRisk))
    //         ));
    //     });
    //
    //     client.publish(&signed_packet).await.unwrap();
    //
    //     handle.await.unwrap()
    // }
}
