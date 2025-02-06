//! Pkarr client for publishing and resolving [SignedPacket]s over [mainline] and/or [Relays](https://pkarr.org/relays).

macro_rules! cross_debug {
    ($($arg:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        log::debug!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::debug!($($arg)*);
    };
}

pub mod cache;

#[cfg(not(wasm_browser))]
pub mod blocking;
pub mod builder;
#[cfg(relays)]
mod relays;

#[cfg(all(test, not(wasm_browser)))]
mod tests;
#[cfg(all(test, wasm_browser))]
mod tests_web;

use futures_lite::{Stream, StreamExt};
use pubky_timestamp::Timestamp;
use std::pin::Pin;
use std::sync::Arc;
use std::{hash::Hash, num::NonZeroUsize};

#[cfg(dht)]
use mainline::{errors::PutMutableError, Dht};

use builder::{ClientBuilder, Config};

#[cfg(relays)]
use crate::client::relays::RelaysClient;
use crate::{Cache, CacheKey, InMemoryCache};
use crate::{PublicKey, SignedPacket};

#[derive(Debug)]
pub(crate) struct Inner {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Option<Arc<dyn Cache>>,
    #[cfg(dht)]
    dht: Option<Dht>,
    #[cfg(relays)]
    relays: Option<RelaysClient>,
    #[cfg(feature = "endpoints")]
    pub(crate) max_recursion_depth: u8,
}

/// Pkarr client for publishing and resolving [SignedPacket]s over
/// [mainline] Dht and/or [Relays](https://pkarr.org/relays).
#[derive(Clone, Debug)]
pub struct Client(pub(crate) Arc<Inner>);

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

        cross_debug!("Starting Pkarr Client {:?}", config);

        #[cfg(dht)]
        let dht = if let Some(builder) = config.dht {
            Some(builder.build().map_err(BuildError::DhtBuildError)?)
        } else {
            None
        };
        #[cfg(not(dht))]
        let dht: Option<()> = None;

        #[cfg(relays)]
        let relays = if let Some(ref relays) = config.relays {
            if relays.is_empty() {
                return Err(BuildError::EmptyListOfRelays);
            }

            let relays_client = RelaysClient::new(
                relays.clone().into_boxed_slice(),
                #[cfg(not(wasm_browser))]
                config.request_timeout,
            );

            Some(relays_client)
        } else {
            None
        };
        #[cfg(not(relays))]
        let relays: Option<()> = None;

        if dht.is_none() && relays.is_none() {
            return Err(BuildError::NoNetwork);
        }

        let client = Client(Arc::new(Inner {
            minimum_ttl: config.minimum_ttl.min(config.maximum_ttl),
            maximum_ttl: config.maximum_ttl.max(config.minimum_ttl),
            cache,
            #[cfg(dht)]
            dht,
            #[cfg(relays)]
            relays,
            #[cfg(feature = "endpoints")]
            max_recursion_depth: config.max_recursion_depth,
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
    #[cfg(dht)]
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
    /// 1. [QueryError]: when the query fails, and you need to retry or debug the network.
    /// 2. [ConcurrencyError]: when an write conflict (or the risk of it) is detedcted.
    ///
    /// If you get a [ConcurrencyError]; you should resolver the most recent packet again,
    /// and repeat the steps in the previous example.
    pub async fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        #[cfg(not(wasm_browser))]
        {
            async_compat::Compat::new(self.publish_inner(signed_packet, cas)).await
        }
        #[cfg(wasm_browser)]
        self.publish_inner(signed_packet, cas).await
    }

    // === Resolve ===

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receieves.
    ///
    /// If you want to get the most recent version of a [SignedPacket],
    /// you should use [Self::resolve_most_recent].
    pub async fn resolve(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        #[cfg(not(wasm_browser))]
        {
            async_compat::Compat::new(self.resolve_inner(public_key)).await
        }
        #[cfg(wasm_browser)]
        self.resolve_inner(public_key).await
    }

    /// Returns the most recent [SignedPacket] found after querying all
    /// [mainline] Dht nodes and or [Relays](https:://pkarr.org/relays).
    ///
    /// Useful if you want to read the most recent packet before publishing
    /// a new packet.
    ///
    /// This is a best effort, and doesn't guarantee consistency.
    pub async fn resolve_most_recent(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        let cache_key: CacheKey = public_key.as_ref().into();

        let cached_packet = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key));

        let mut stream = self.resolve_stream(
            public_key.clone(),
            cache_key,
            cached_packet.map(|s| s.timestamp()),
        );
        while stream.next().await.is_some() {}

        self.cache().and_then(|cache| cache.get(&public_key.into()))
    }

    // === Private Methods ===

    async fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        let cache_key: CacheKey = signed_packet.public_key().into();

        // Check conflict
        if let Some(cached) = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key))
        {
            if cached.more_recent_than(signed_packet) {
                return Err(ConcurrencyError::NotMostRecent)?;
            }
        } else if let Some(cas) = cas {
            if let Some(cached) = self
                .cache()
                .as_ref()
                .and_then(|cache| cache.get(&cache_key))
            {
                if cached.timestamp() != cas {
                    return Err(ConcurrencyError::CasFailed)?;
                }
            }
        }

        if let Some(cache) = self.cache() {
            cache.put(&cache_key, signed_packet);
        }

        self.select_publish_future(signed_packet, cas).await
    }

    /// Returns the first result from either the DHT or the Relays client or both.
    async fn select_publish_future(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        // Handle DHT and Relay futures based on feature flags and target family
        #[cfg(dht)]
        let dht_future = self.dht().map(|node| async {
            node.as_async()
                .put_mutable(signed_packet.into(), cas.map(|t| t.as_u64() as i64))
                .await
                .map(|_| Ok(()))?
        });

        #[cfg(relays)]
        let relays_future = self
            .0
            .relays
            .as_ref()
            .map(|relays| relays.publish(signed_packet, cas));

        #[cfg(all(dht, not(relays)))]
        return dht_future.expect("infallible").await;

        #[cfg(all(relays, not(dht)))]
        return relays_future.expect("infallible").await;

        #[cfg(all(dht, relays))]
        return if dht_future.is_some() && relays_future.is_some() {
            futures_lite::future::or(
                dht_future.expect("infallible"),
                relays_future.expect("infallible"),
            )
            .await
        } else if dht_future.is_some() {
            dht_future.expect("infallible").await
        } else {
            relays_future.expect("infallible").await
        };
    }

    pub(crate) async fn resolve_inner(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        let public_key = public_key.clone();

        let cache_key: CacheKey = public_key.as_ref().into();

        let cached_packet = self
            .cache()
            .as_ref()
            .and_then(|cache| cache.get(&cache_key));

        // Stream is a future, so it won't run until we await or spawn it.
        let mut stream = self.resolve_stream(
            public_key,
            cache_key,
            cached_packet.as_ref().map(|s| s.timestamp()),
        );

        if let Some(cached_packet) = cached_packet {
            if cached_packet.is_expired(self.0.minimum_ttl, self.0.maximum_ttl) {
                #[cfg(not(wasm_browser))]
                tokio::spawn(async move { while stream.next().await.is_some() {} });
                #[cfg(wasm_browser)]
                wasm_bindgen_futures::spawn_local(
                    async move { while stream.next().await.is_some() {} },
                );
            }

            cross_debug!(
                "responding with cached packet even if expired. public_key: {}",
                cached_packet.public_key()
            );
        } else {
            // Wait for the earliest positive response.
            let _ = stream.next().await;
        };

        self.cache().and_then(|cache| cache.get(&cache_key))
    }

    #[cfg(wasm_browser)]
    fn resolve_stream(
        &self,
        public_key: PublicKey,
        cache_key: CacheKey,
        more_recent_than: Option<Timestamp>,
    ) -> Pin<Box<dyn Stream<Item = SignedPacket>>> {
        let cache = self.0.cache.clone();

        let stream = self
            .0
            .relays
            .as_ref()
            .expect("infallible")
            .resolve_futures(&public_key, more_recent_than)
            .filter_map(|opt| opt)
            .filter_map(move |signed_packet| {
                filter_incoming_signed_packet(&public_key, cache.clone(), &cache_key, signed_packet)
            });

        Box::pin(stream)
    }

    #[cfg(not(wasm_browser))]
    /// Returns a [Stream] of incoming [SignedPacket]s.
    fn resolve_stream(
        &self,
        public_key: PublicKey,
        cache_key: CacheKey,
        more_recent_than: Option<Timestamp>,
    ) -> Pin<Box<dyn Stream<Item = SignedPacket> + Send>> {
        let cache = self.0.cache.clone();

        self.merged_resolve_stream(&public_key, more_recent_than)
            .filter_map(move |signed_packet| {
                filter_incoming_signed_packet(&public_key, cache.clone(), &cache_key, signed_packet)
            })
            .boxed()
    }

    #[cfg(not(wasm_browser))]
    /// Returns a Stream from both the DHT and Relays client.
    fn merged_resolve_stream(
        &self,
        public_key: &PublicKey,
        more_recent_than: Option<Timestamp>,
    ) -> Pin<Box<dyn Stream<Item = SignedPacket> + Send>> {
        #[cfg(dht)]
        let dht_stream = match self.dht() {
            Some(node) => map_dht_stream(node.as_async().get_mutable(
                public_key.as_bytes(),
                None,
                more_recent_than.map(|t| t.as_u64() as i64),
            )),
            None => None,
        };

        #[cfg(relays)]
        let relays_stream = self
            .0
            .relays
            .as_ref()
            .map(|relays| relays.resolve(public_key, more_recent_than));

        #[cfg(all(dht, not(relays)))]
        return dht_stream.expect("infallible");

        #[cfg(all(relays, not(dht)))]
        return relays_stream.expect("infallible");

        #[cfg(all(dht, relays))]
        Box::pin(match (dht_stream, relays_stream) {
            (Some(s), None) | (None, Some(s)) => s,
            (Some(a), Some(b)) => Box::pin(futures_lite::stream::or(a, b)),
            (None, None) => unreachable!("should not create a client with no network"),
        })
    }
}

fn filter_incoming_signed_packet(
    public_key: &PublicKey,
    cache: Option<Arc<dyn Cache>>,
    cache_key: &CacheKey,
    signed_packet: SignedPacket,
) -> Option<SignedPacket> {
    let new_packet: Option<SignedPacket> = if let Some(cached) = cache
        .clone()
        .and_then(|cache| cache.clone().get_read_only(cache_key))
    {
        if signed_packet.more_recent_than(&cached) {
            cross_debug!("Received more recent packet than in cache. public_key: {public_key}",);

            Some(signed_packet)
        } else {
            None
        }
    } else {
        cross_debug!("Received new packet after cache miss. public_key: {public_key}");

        Some(signed_packet)
    };

    if let Some(packet) = new_packet {
        if let Some(cache) = &cache {
            cache.put(cache_key, &packet)
        };

        Some(packet)
    } else {
        None
    }
}

#[cfg(dht)]
fn map_dht_stream(
    stream: mainline::async_dht::GetStream<mainline::MutableItem>,
) -> Option<Pin<Box<dyn Stream<Item = SignedPacket> + Send>>> {
    return Some(
        stream
            .filter_map(
                move |mutable_item| match SignedPacket::try_from(mutable_item) {
                    Ok(signed_packet) => Some(signed_packet),
                    Err(error) => {
                        cross_debug!("Got an invalid signed packet from the DHT. Error: {error}");
                        None
                    }
                },
            )
            .boxed(),
    );
}

#[derive(thiserror::Error, Debug)]
/// Errors occuring during building a [Client]
pub enum BuildError {
    #[error("Client configured without Mainline node or relays.")]
    /// Client configured without Mainline node or relays.
    NoNetwork,

    #[error("Failed to build the Dht client {0}")]
    /// Failed to build the Dht client.
    DhtBuildError(std::io::Error),

    #[error("Passed an empty list of relays")]
    /// Passed an empty list of relays
    EmptyListOfRelays,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
/// Errors occuring during publishing a [SignedPacket]
pub enum PublishError {
    #[error(transparent)]
    Query(#[from] QueryError),

    #[error(transparent)]
    Concurrency(#[from] ConcurrencyError),

    // === Relays only errors ===
    #[error("All relays responded with unexpected responses, check debug logs.")]
    /// All relays responded with unexpected responses, check debug logs.
    UnexpectedResponses,
}

#[derive(thiserror::Error, Debug, Clone)]
/// Errors that requires either a retry or debugging the network condition.
pub enum QueryError {
    /// Publish query timed out with no responses neither success or errors, from Dht or relays.
    #[error("Publish query timed out with no responses neither success or errors.")]
    Timeout,

    #[error("Publishing SignedPacket to Mainline failed.")]
    /// Publishing SignedPacket to Mainline failed.
    NoClosestNodes,

    #[error("Publishing SignedPacket to Mainline failed code: {0}, description: {1}.")]
    /// Publishing SignedPacket to Mainline failed, received an error response.
    DhtErrorResponse(i32, String),

    #[error("Most relays responded with bad request")]
    /// Most relays responded with bad request
    BadRequest,
}

impl Eq for QueryError {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialEq for QueryError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(dht)]
            (
                QueryError::DhtErrorResponse(self_error, _),
                QueryError::DhtErrorResponse(other_error, _),
            ) => self_error == other_error,
            (s, o) => s == o,
        }
    }
}

impl Hash for QueryError {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            QueryError::Timeout => 0.hash(state),
            QueryError::NoClosestNodes => 1.hash(state),
            QueryError::DhtErrorResponse(code, _) => {
                let mut bytes = vec![2];
                bytes.extend_from_slice(&code.to_be_bytes());

                state.write(bytes.as_slice());
            }
            QueryError::BadRequest => 3.hash(state),
        }
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
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

#[cfg(dht)]
impl From<PutMutableError> for PublishError {
    fn from(value: PutMutableError) -> Self {
        match value {
            PutMutableError::Query(error) => PublishError::Query(match error {
                mainline::errors::PutQueryError::Timeout => QueryError::Timeout,
                mainline::errors::PutQueryError::NoClosestNodes => QueryError::NoClosestNodes,
                mainline::errors::PutQueryError::ErrorResponse(error) => {
                    QueryError::DhtErrorResponse(error.code, error.description)
                }
            }),
            PutMutableError::Concurrency(error) => PublishError::Concurrency(match error {
                mainline::errors::ConcurrencyError::ConflictRisk => ConcurrencyError::ConflictRisk,
                mainline::errors::ConcurrencyError::NotMostRecent => {
                    ConcurrencyError::NotMostRecent
                }
                mainline::errors::ConcurrencyError::CasFailed => ConcurrencyError::CasFailed,
            }),
        }
    }
}
