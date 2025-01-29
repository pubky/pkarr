//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or [Relays](https://pkarr.org/relays).

use flume::r#async::RecvStream;
use flume::Receiver;
use futures_lite::{Stream, StreamExt};
use pubky_timestamp::Timestamp;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::debug;

#[cfg(feature = "dht")]
use mainline::{
    errors::{DhtWasShutdown, PutMutableError},
    Dht,
};

use crate::{Cache, CacheKey, InMemoryCache};
use crate::{PublicKey, SignedPacket};

mod builder;
// #[cfg(feature = "relays")]
// mod relays;

pub use builder::{ClientBuilder, Config};

#[derive(Clone, Debug)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or
/// [Relays](https://pkarr.org/relays).
pub struct Inner {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Option<Box<dyn Cache>>,
    #[cfg(feature = "dht")]
    dht: Option<Dht>,
}

#[derive(Clone, Debug)]
pub struct Client(Arc<Inner>);

impl Client {
    pub fn new(config: Config) -> Result<Client, BuildError> {
        #[cfg(feature = "relays")]
        let relays_client = if let Some(ref relays) = config.relays {
            if relays.is_empty() {
                return Err(BuildError::EmptyListOfRelays);
            }

            Some(())
        } else {
            None
        };

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

        debug!(?config, "Starting Pkarr Client..");

        let dht = if let Some(builder) = config.dht {
            Some(builder.build().map_err(BuildError::DhtBuildError)?)
        } else {
            None
        };

        if dht.is_none() && relays_client.is_none() {
            return Err(BuildError::NoNetwork);
        }

        let client = Client(Arc::new(Inner {
            minimum_ttl: config.minimum_ttl,
            maximum_ttl: config.maximum_ttl,
            cache,
            #[cfg(feature = "dht")]
            dht,
        }));

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

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> Option<&dyn Cache> {
        self.0.cache.as_deref()
    }

    /// Returns a reference to the internal [mainline::Dht] node.
    ///
    /// Gives you access to methods like [mainline::Dht::info],
    /// [mainline::Dht::bootstrapped], and [mainline::Dht::to_bootstrap]
    /// among ther rest of the API.
    #[cfg(feature = "dht")]
    pub fn dht(&self) -> Option<mainline::Dht> {
        self.0.dht.as_ref().cloned()
    }

    // === Publish ===

    /// Publishes a [SignedPacket] to the [mainline] Dht and or [Relays](https://pkarr.org/relays).
    ///
    /// # Lost Update Problem
    ///
    /// Mainline DHT and remote relays form a distributed network, and like all distributed networks,
    /// it is vulnerable to [Write–write conflict](https://en.wikipedia.org/wiki/Write-write_conflict).
    ///
    /// ## Read first
    ///
    /// To mitigate the risk of lost updates, you should call the [Self::resolve_most_recent] method
    /// then start authoring the new [SignedPacket] based on the most recent as in the following example:
    ///
    ///```rust
    /// use mainline::Testnet;
    /// use pkarr::{Client, SignedPacket, Keypair};
    ///
    /// #[tokio::main]
    /// async fn run() -> anyhow::Result<()> {
    ///     let testnet = Testnet::new(3)?;
    ///     let client = Client::builder()
    ///         // Disable the default network settings (builtin relays and mainline bootstrap nodes).
    ///         .no_default_network()
    ///         .bootstrap(&testnet.bootstrap)
    ///         .build()?;
    ///
    ///     let keypair = Keypair::random();
    ///
    ///     let (signed_packet, cas) = if let Some(most_recent) = client
    ///         .resolve_most_recent(&keypair.public_key()).await?
    ///     {
    ///
    ///         let mut builder = SignedPacket::builder();
    ///
    ///         // 1. Optionally inherit all or some of the existing records.
    ///         for record in most_recent.all_resource_records() {
    ///             let name = record.name.to_string();
    ///
    ///             if name != "foo" && name != "sercert" {
    ///                 builder = builder.record(record.clone());
    ///             }
    ///         };
    ///
    ///         // 2. Optionally add more new records.
    ///         let signed_packet = builder
    ///             .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///             .a("secret".try_into()?, 42.into(), 30)
    ///             .sign(&keypair)?;
    ///
    ///         (
    ///             signed_packet,
    ///             // 3. Use the most recent [SignedPacket::timestamp] as a `CAS`.
    ///             Some(most_recent.timestamp())
    ///         )
    ///     } else {
    ///         (
    ///             SignedPacket::builder()
    ///                 .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///                 .a("secret".try_into()?, 42.into(), 30)
    ///                 .sign(&keypair)?,
    ///             None
    ///         )
    ///     };
    ///
    ///     client.publish(&signed_packet, cas).await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Errors
    ///
    /// This method may return on of these errors:
    ///
    /// 1. [Shutdown][PublishError::ClientWasShutdown].
    /// 2. [QueryError]: when the query fails, and you need to retry or debug the network.
    /// 3. [ConcurrencyError]: when an write conflict (or the risk of it) is detedcted.
    ///
    /// If you get a [ConcurrencyError]; you should resolver the most recent packet again,
    /// and repeat the steps in the previous example.
    pub async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let cache_key: CacheKey = signed_packet.public_key().into();

        self.check_conflict(signed_packet, &cache_key, cas)?;

        if let Some(cache) = self.cache() {
            cache.put(&cache_key, signed_packet);
        }

        if let Some(node) = self.dht() {
            node.as_async()
                .put_mutable(signed_packet.into(), cas.map(|t| t.as_u64() as i64))
                .await?;
        }

        Ok(())
    }

    /// Publishes a [SignedPacket] to the [mainline] Dht and or [Relays](https://pkarr.org/relays).
    ///
    /// # Lost Update Problem
    ///
    /// Mainline DHT and remote relays form a distributed network, and like all distributed networks,
    /// it is vulnerable to [Write–write conflict](https://en.wikipedia.org/wiki/Write-write_conflict).
    ///
    /// ## Read first
    ///
    /// To mitigate the risk of lost updates, you should call the [Self::resolve_most_recent] method
    /// then start authoring the new [SignedPacket] based on the most recent as in the following example:
    ///
    ///```rust
    /// use mainline::Testnet;
    /// use pkarr::{Client, SignedPacket, Keypair};
    ///
    /// fn run() -> anyhow::Result<()> {
    ///     let testnet = Testnet::new(3)?;
    ///     let client = Client::builder()
    ///         // Disable the default network settings (builtin relays and mainline bootstrap nodes).
    ///         .no_default_network()
    ///         .bootstrap(&testnet.bootstrap)
    ///         .build()?;
    ///
    ///     let keypair = Keypair::random();
    ///
    ///     let (signed_packet, cas) = if let Some(most_recent) = client
    ///         .resolve_most_recent_sync(&keypair.public_key())?
    ///     {
    ///
    ///         let mut builder = SignedPacket::builder();
    ///
    ///         // 1. Optionally inherit all or some of the existing records.
    ///         for record in most_recent.all_resource_records() {
    ///             let name = record.name.to_string();
    ///
    ///             if name != "foo" && name != "sercert" {
    ///                 builder = builder.record(record.clone());
    ///             }
    ///         };
    ///
    ///         // 2. Optionally add more new records.
    ///         let signed_packet = builder
    ///             .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///             .a("secret".try_into()?, 42.into(), 30)
    ///             .sign(&keypair)?;
    ///
    ///         (
    ///             signed_packet,
    ///             // 3. Use the most recent [SignedPacket::timestamp] as a `CAS`.
    ///             Some(most_recent.timestamp())
    ///         )
    ///     } else {
    ///         (
    ///             SignedPacket::builder()
    ///                 .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///                 .a("secret".try_into()?, 42.into(), 30)
    ///                 .sign(&keypair)?,
    ///             None
    ///         )
    ///     };
    ///
    ///     client.publish_sync(&signed_packet, cas)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Errors
    ///
    /// This method may return on of these errors:
    ///
    /// 1. [Shutdown][PublishError::ClientWasShutdown].
    /// 2. [QueryError]: when the query fails, and you need to retry or debug the network.
    /// 3. [ConcurrencyError]: when an write conflict (or the risk of it) is detedcted.
    ///
    /// If you get a [ConcurrencyError]; you should resolver the most recent packet again,
    /// and repeat the steps in the previous example.
    pub fn publish_sync(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let cache_key: CacheKey = signed_packet.public_key().into();

        self.check_conflict(signed_packet, &cache_key, cas)?;

        if let Some(cache) = self.cache() {
            cache.put(&cache_key, signed_packet);
        }

        if let Some(node) = self.dht() {
            node.put_mutable(signed_packet.into(), cas.map(|t| t.as_u64() as i64))?;
        }

        Ok(())
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
    // TODO: Can we shutdown request? can we merge
    pub async fn shutdown(&mut self) {
        if let Some(node) = self.dht() {
            node.as_async().shutdown().await;
        }
    }

    /// Shutdown the actor thread loop.
    pub fn shutdown_sync(&self) {
        if let Some(mut node) = self.dht() {
            node.shutdown();
        }
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

        let cache_key: CacheKey = public_key.into();

        let cached_packet = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key));

        // Sending the `timestamp` of the known cache, help save some bandwith,
        // since remote nodes will not send the encoded packet if they don't know
        // any more recent versions.
        let most_recent_known_timestamp = cached_packet
            .as_ref()
            .map(|cached| cached.timestamp().as_u64());

        // TODO: use configured min/max ttl
        // Should query?
        if cached_packet
            .as_ref()
            .map(|c| c.is_expired(self.0.minimum_ttl, self.0.maximum_ttl))
            .unwrap_or(true)
        {
            debug!(
                ?public_key,
                "querying the DHT to hydrate our cache for later."
            );

            if let Some(node) = self.dht() {
                let iter = node.get_mutable(
                    public_key.as_bytes(),
                    None,
                    most_recent_known_timestamp.map(|t| t as i64),
                )?;

                let tx = tx.clone();
                let public_key = public_key.clone();
                let cache = self.0.cache.clone();

                // TODO: Avoid a new thread for every request.
                std::thread::spawn(move || {
                    for mutable_item in iter {
                        match SignedPacket::try_from(mutable_item) {
                            Ok(signed_packet) => {
                                let new_packet: Option<SignedPacket> = if let Some(ref cached) =
                                    cache
                                        .as_ref()
                                        .and_then(|cache| cache.get_read_only(&cache_key))
                                {
                                    if signed_packet.more_recent_than(cached) {
                                        debug!(
                                            ?public_key,
                                            "Received more recent packet than in cache"
                                        );

                                        Some(signed_packet)
                                    } else {
                                        None
                                    }
                                } else {
                                    debug!(?public_key, "Received new packet after cache miss");
                                    Some(signed_packet)
                                };

                                if let Some(packet) = new_packet {
                                    if let Some(cache) = &cache {
                                        cache.put(&cache_key, &packet)
                                    };

                                    let _ = tx.send(packet);
                                }
                            }
                            Err(error) => {
                                debug!(?error, "Got an invalid signed packet from the DHT");
                            }
                        }
                    }
                });
            }
        }

        if let Some(cached_packet) = cached_packet {
            debug!(
                public_key = ?cached_packet.public_key(),
                "responding with cached packet even if expired"
            );

            let _ = tx.send(cached_packet);
        }

        Ok(rx)
    }

    fn check_conflict(
        &self,
        signed_packet: &SignedPacket,
        cache_key: &CacheKey,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        if let Some(cached) = self.cache().as_ref().and_then(|cache| cache.get(cache_key)) {
            if cached.timestamp() >= signed_packet.timestamp() {
                return Err(ConcurrencyError::NotMostRecent)?;
            }
        } else if let Some(cas) = cas {
            if let Some(cached) = self.cache().as_ref().and_then(|cache| cache.get(cache_key)) {
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
    #[error("Client configured without Mainline node or relays.")]
    /// Client configured without Mainline node or relays.
    NoNetwork,

    #[cfg(feature = "dht")]
    #[error("Failed to build the Dht client {0}")]
    /// Failed to build the Dht client.
    DhtBuildError(std::io::Error),

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

impl From<DhtWasShutdown> for ClientWasShutdown {
    fn from(_: DhtWasShutdown) -> Self {
        Self
    }
}

#[derive(thiserror::Error, Debug, Clone)]
/// Errors occuring during publishing a [SignedPacket]
pub enum PublishError {
    #[error(transparent)]
    Query(QueryError),

    #[error(transparent)]
    Concurrency(#[from] ConcurrencyError),

    #[error("Client was shutdown")]
    ClientWasShutdown,
}

#[derive(thiserror::Error, Debug, Clone)]
/// Errors that requires either a retry or debugging the network condition.
pub enum QueryError {
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
    DhtErrorResponse(mainline::rpc::messages::ErrorSpecific),
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

impl From<PutMutableError> for PublishError {
    fn from(value: PutMutableError) -> Self {
        match value {
            PutMutableError::Shutdown(_) => PublishError::ClientWasShutdown,
            PutMutableError::Query(error) => PublishError::Query(match error {
                mainline::rpc::PutQueryError::Timeout => QueryError::Timeout,
                mainline::rpc::PutQueryError::NoClosestNodes => QueryError::NoClosestNodes,
                mainline::rpc::PutQueryError::ErrorResponse(error) => {
                    QueryError::DhtErrorResponse(error)
                }
            }),
            PutMutableError::Concurrency(error) => PublishError::Concurrency(match error {
                mainline::rpc::ConcurrencyError::ConflictRisk => ConcurrencyError::ConflictRisk,
                mainline::rpc::ConcurrencyError::NotMostRecent => ConcurrencyError::NotMostRecent,
                mainline::rpc::ConcurrencyError::CasFailed => ConcurrencyError::CasFailed,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Combined client tests

    use std::{thread, time::Duration};

    use native::{BuildError, ConcurrencyError, PublishError};
    use pkarr_relay::Relay;
    use rstest::rstest;

    use super::super::*;
    use crate::{Keypair, SignedPacket};

    #[derive(Copy, Clone)]
    enum Networks {
        Dht,
        #[cfg(feature = "relays")]
        Relays,
        Both,
    }

    /// Parametric [ClientBuilder] with no default networks,
    /// instead it uses mainline or relays depending on `networks` enum.
    fn builder(relay: &Relay, testnet: &mainline::Testnet, networks: Networks) -> ClientBuilder {
        let mut builder = Client::builder();

        builder
            .no_default_network()
            // Because of pkarr_relay crate, dht is always enabled.
            .bootstrap(&testnet.bootstrap)
            .resolvers(Some(vec![relay.resolver_address().to_string()]))
            .request_timeout(Duration::from_millis(100));

        match networks {
            Networks::Dht => {}
            #[cfg(feature = "relays")]
            Networks::Relays => {
                builder
                    .no_default_network()
                    .relays(Some(vec![relay.local_url()]));
            }
            Networks::Both => {
                #[cfg(feature = "relays")]
                {
                    builder.relays(Some(vec![relay.local_url()]));
                }
            }
        }

        dbg!(&builder);

        builder
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    // TODO: enable testing relays only
    // #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn publish_resolve(#[case] networks: Networks) {
        let (relay, testnet) = Relay::start_test().await.unwrap();

        let a = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet, None).await.unwrap();

        let b = builder(&relay, &testnet, networks).build().unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    // #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn send(#[case] networks: Networks) {
        let (relay, testnet) = Relay::start_test().await.unwrap();

        let a = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet, None).await.unwrap();

        let b = builder(&relay, &testnet, networks).build().unwrap();

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

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    // #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn return_expired_packet_fallback(#[case] networks: Networks) {
        let (relay, testnet) = Relay::start_test().await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client
            .cache()
            .unwrap()
            .put(&keypair.public_key().into(), &signed_packet);

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, Some(signed_packet));
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    // #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn ttl_0_test(#[case] networks: Networks) {
        let (relay, testnet) = Relay::start_test().await.unwrap();

        let client = builder(&relay, &testnet, networks)
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

        client.publish(&signed_packet, None).await.unwrap();

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

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    // #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn not_found(#[case] networks: Networks) {
        let (relay, testnet) = Relay::start_test().await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert_eq!(resolved, None);
    }

    #[test]
    fn no_network() {
        assert!(matches!(
            Client::builder().no_default_network().build(),
            Err(BuildError::NoNetwork)
        ));
    }
}
