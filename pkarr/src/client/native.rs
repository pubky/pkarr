//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or [Relays](https://pkarr.org/relays).

use flume::r#async::RecvStream;
use flume::Receiver;
use futures_lite::{stream, Stream, StreamExt};
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

use super::builder::Config;

use crate::client::relays::RelaysClient;
use crate::{Cache, CacheKey, InMemoryCache};
use crate::{PublicKey, SignedPacket};

pub use super::builder::ClientBuilder;

#[derive(Debug)]
pub struct Inner {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Option<Arc<dyn Cache>>,
    #[cfg(feature = "dht")]
    dht: Option<Dht>,
    #[cfg(feature = "relays")]
    relays: Option<RelaysClient>,
}

/// Pkarr client for publishing and resolving [SignedPacket]s over
/// [mainline] Dht and/or [Relays](https://pkarr.org/relays).
#[derive(Clone, Debug)]
pub struct Client(Arc<Inner>);

impl Client {
    pub(crate) fn new(config: Config) -> Result<Client, BuildError> {
        let cache = if config.cache_size == 0 {
            None
        } else {
            Some(
                config.cache.clone().unwrap_or(Arc::new(InMemoryCache::new(
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

        #[cfg(feature = "relays")]
        let relays = if let Some(ref relays) = config.relays {
            if relays.is_empty() {
                return Err(BuildError::EmptyListOfRelays);
            }

            let relays_client =
                RelaysClient::new(relays.clone().into_boxed_slice(), config.request_timeout);

            Some(relays_client)
        } else {
            None
        };

        if dht.is_none() && relays.is_none() {
            return Err(BuildError::NoNetwork);
        }

        let client = Client(Arc::new(Inner {
            minimum_ttl: config.minimum_ttl.min(config.maximum_ttl),
            maximum_ttl: config.maximum_ttl.max(config.minimum_ttl),
            cache,
            #[cfg(feature = "dht")]
            dht,
            #[cfg(feature = "relays")]
            relays,
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
        async_compat::Compat::new(async {
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

            // TODO: support other modes than DhtFirst

            if let Some(ref relays) = self.0.relays {
                relays
                    .publish(signed_packet, cas)
                    .recv_async()
                    .await
                    .expect("pkarr relays publish dropped sender too soon")?;
            }

            Ok(())
        })
        .await
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
        futures_lite::future::block_on(self.publish(signed_packet, cas))
    }

    // === Resolve ===

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to get the most recent version of a [SignedPacket],
    /// you should use [Self::resolve_most_recent].
    pub async fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        let public_key = public_key.clone();

        let cache_key: CacheKey = public_key.as_ref().into();

        let cached_packet = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key));

        if let Some(cached_packet) = cached_packet {
            if cached_packet.is_expired(self.0.minimum_ttl, self.0.maximum_ttl) {
                let stream =
                    self.resolve_stream(public_key, cache_key, Some(cached_packet.timestamp()))?;

                tokio::spawn(async move {
                    let mut stream = stream;
                    while stream.next().await.is_some() {}
                });
            }

            debug!(
                public_key = ?cached_packet.public_key(),
                "responding with cached packet even if expired"
            );

            Ok(Some(cached_packet))
        } else {
            // We have nothing locally, we have to resolve.
            Ok(self
                .resolve_stream(public_key, cache_key, None)?
                .next()
                .await)
        }
    }

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to get the most recent version of a [SignedPacket],
    /// you should use [Self::resolve_most_recent_sync].
    pub fn resolve_sync(
        &self,
        public_key: &PublicKey,
    ) -> Result<Option<SignedPacket>, ClientWasShutdown> {
        futures_lite::future::block_on(self.resolve(public_key))
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
        let cache_key: CacheKey = public_key.as_ref().into();

        let cached_packet = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key));

        let mut stream = self.resolve_stream(
            public_key.clone(),
            cache_key,
            cached_packet.map(|s| s.timestamp()),
        )?;
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
        futures_lite::future::block_on(self.resolve_most_recent(public_key))
    }

    // === Private Methods ===

    /// Returns a [Stream] of incoming [SignedPacket]s.
    pub fn resolve_stream(
        &self,
        public_key: PublicKey,
        cache_key: CacheKey,
        most_recent_known_timestamp: Option<Timestamp>,
    ) -> Result<Pin<Box<dyn Stream<Item = SignedPacket> + Send>>, ClientWasShutdown> {
        let dht_stream = match self.dht() {
            Some(node) => map_dht_stream(node.as_async().get_mutable(
                public_key.as_bytes(),
                None,
                most_recent_known_timestamp.map(|t| t.as_u64() as i64),
            )?),
            None => None,
        };

        let relays_stream = self
            .0
            .relays
            .as_ref()
            .map(|relays| relays.resolve(&public_key));

        let cache = self.0.cache.clone();

        let stream = match (dht_stream, relays_stream) {
            (Some(s), None) | (None, Some(s)) => s,
            (Some(a), Some(b)) => Box::pin(stream::or(a, b)),
            (None, None) => unreachable!("should not create a client with no network"),
        }
        .map(move |signed_packet| {
            let new_packet: Option<SignedPacket> = if let Some(cached) = cache
                .clone()
                .and_then(|cache| cache.clone().get_read_only(&cache_key))
            {
                if signed_packet.more_recent_than(&cached) {
                    debug!(?public_key, "Received more recent packet than in cache");

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

                Some(packet)
            } else {
                None
            }
        })
        .filter_map(|x| x);

        Ok(Box::pin(stream))
    }

    fn check_conflict(
        &self,
        signed_packet: &SignedPacket,
        cache_key: &CacheKey,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        if let Some(cached) = self.cache().as_ref().and_then(|cache| cache.get(cache_key)) {
            if cached.more_recent_than(signed_packet) {
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

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(ref mut dht) = self.dht {
            dht.shutdown();
        };
    }
}

#[cfg(feature = "dht")]
fn map_dht_stream(
    stream: mainline::async_dht::GetStream<mainline::MutableItem>,
) -> Option<Pin<Box<dyn Stream<Item = SignedPacket> + Send>>> {
    return Some(
        stream
            .map(
                move |mutable_item| match SignedPacket::try_from(mutable_item) {
                    Ok(signed_packet) => Some(signed_packet),
                    Err(error) => {
                        debug!(?error, "Got an invalid signed packet from the DHT");
                        None
                    }
                },
            )
            .filter_map(|x| x)
            .boxed(),
    );
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
    use pubky_timestamp::Timestamp;
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

        builder
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn publish_resolve(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

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
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn client_send(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let a = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        dbg!("BEFORE PUBLISH");
        a.publish(&signed_packet, None).await.unwrap();
        dbg!("AFTER PUBLISH");

        let b = builder(&relay, &testnet, networks).build().unwrap();

        tokio::spawn(async move {
            dbg!("RESOLVING");
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
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn return_expired_packet_fallback(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

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
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn ttl_0_test(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks)
            .maximum_ttl(0)
            .build()
            .unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

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
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn not_found(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

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

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn repeated_publish_query(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let id = client.publish(&signed_packet, None).await.unwrap();

        assert_eq!(client.publish(&signed_packet, None).await.unwrap(), id);
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn concurrent_resolve(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let a = builder(&relay, &testnet, networks).build().unwrap();
        let b = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet, None).await.unwrap();

        let public_key = signed_packet.public_key();
        let bclone = b.clone();
        let _stream = tokio::spawn(async move { bclone.resolve(&public_key).await.unwrap() });

        let response_second = b
            .resolve(&signed_packet.public_key())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(&response_second.as_bytes(), &signed_packet.as_bytes());

        assert!(_stream.await.is_ok())
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn concurrent_publish_same_packet(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        let mut handles = vec![];

        for _ in 0..2 {
            let client = client.clone();
            let signed_packet = signed_packet.clone();

            handles.push(tokio::spawn(async move {
                client.publish(&signed_packet, None).await.unwrap()
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn concurrent_publish_of_different_packets(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let mut handles = vec![];

        let keypair = Keypair::random();

        let timestamp = Timestamp::now();

        for i in 0..2 {
            let client = client.clone();

            let signed_packet = SignedPacket::builder()
                .txt(
                    format!("foo{i}").as_str().try_into().unwrap(),
                    "bar".try_into().unwrap(),
                    30,
                )
                .timestamp(timestamp)
                .sign(&keypair)
                .unwrap();

            handles.push(tokio::spawn(async move {
                let result = client.publish(&signed_packet, None).await;

                if i == 0 {
                    assert!(matches!(result, Ok(_)))
                } else {
                    assert!(matches!(
                        result,
                        Err(PublishError::Concurrency(ConcurrencyError::ConflictRisk))
                    ))
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn concurrent_publish_different_with_cas(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        // First
        let cloned_client = client.clone();
        let cloned_keypair = keypair.clone();
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&cloned_keypair)
            .unwrap();

        let handle = tokio::spawn(async move {
            let result = cloned_client.publish(&signed_packet, None).await;

            assert!(matches!(result, Ok(_)))
        });

        // Second
        {
            let signed_packet = SignedPacket::builder()
                .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
                .sign(&keypair)
                .unwrap();

            let most_recent = client
                .resolve_most_recent(&keypair.public_key())
                .await
                .unwrap();

            if let Some(cas) = most_recent.map(|s| s.timestamp()) {
                client.publish(&signed_packet, Some(cas)).await.unwrap();
            } else {
                client.publish(&signed_packet, None).await.unwrap();
            }
        }

        handle.await.unwrap();
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn conflict_302(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet_builder =
            SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);

        let t1 = Timestamp::now();
        let t2 = Timestamp::now();

        client
            .publish(
                &signed_packet_builder
                    .clone()
                    .timestamp(t2)
                    .sign(&keypair)
                    .unwrap(),
                None,
            )
            .await
            .unwrap();

        assert!(matches!(
            client
                .publish(
                    &signed_packet_builder.timestamp(t1).sign(&keypair).unwrap(),
                    None
                )
                .await,
            Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent))
        ));
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn conflict_301_cas(#[case] networks: Networks) {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::start_test(&testnet).await.unwrap();

        let client = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet_builder =
            SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);

        let t1 = Timestamp::now();
        let t2 = Timestamp::now();

        client
            .publish(
                &signed_packet_builder
                    .clone()
                    .timestamp(t2)
                    .sign(&keypair)
                    .unwrap(),
                None,
            )
            .await
            .unwrap();

        assert!(matches!(
            client
                .publish(&signed_packet_builder.sign(&keypair).unwrap(), Some(t1))
                .await,
            Err(PublishError::Concurrency(ConcurrencyError::CasFailed))
        ));
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[test]
    fn no_tokio_sync(#[case] networks: Networks) {
        let (relay, testnet) = futures_lite::future::block_on(async_compat::Compat::new(async {
            let testnet = mainline::Testnet::new(10).unwrap();
            let relay = Relay::start_test(&testnet).await.unwrap();

            (relay, testnet)
        }));

        let a = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish_sync(&signed_packet, None).unwrap();

        let b = builder(&relay, &testnet, networks).build().unwrap();

        let resolved = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve_sync(&keypair.public_key()).unwrap().unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::both_networks(Networks::Both)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[test]
    fn no_tokio_async(#[case] networks: Networks) {
        futures_lite::future::block_on(async {
            let (relay, testnet) = async_compat::Compat::new(async {
                let testnet = mainline::Testnet::new(10).unwrap();
                let relay = Relay::start_test(&testnet).await.unwrap();

                (relay, testnet)
            })
            .await;

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
        });
    }

    // TODO: test background resolve query
    // TODO: test multiple relays
}
