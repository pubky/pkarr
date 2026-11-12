use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::{
    Cache, CacheKey, ClientBuilder, InMemoryCache, PublicKey, ResolvePolicy, SignedPacket,
    StoredNodeCount,
};

use super::backend::{Backend, BackendResolvePolicy, CacheContext};
use super::builder::Config;
use super::errors::ResolveError;
use super::{BuildError, PublishError};

/// Pkarr client for publishing and resolving [`SignedPacket`]s over the configured networks.
#[derive(Clone, Debug)]
pub struct Client {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Option<Arc<dyn Cache>>,
    cache_write_lock: Arc<Mutex<()>>,
    backend: Arc<Backend>,
    #[cfg(feature = "endpoints")]
    pub(crate) max_recursion_depth: u8,
}

impl Client {
    /// Returns a builder for editing the config before creating a `Client`.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub(crate) fn new(config: Config) -> Result<Self, BuildError> {
        let cache = build_cache(config.cache_size, config.cache);

        #[cfg(relays)]
        let relay = config
            .relays
            .map(|relays| {
                Backend::relay(relays, config.relay_request_timeout, config.reqwest_client)
            })
            .transpose()?;
        #[cfg(not(relays))]
        let relay: Option<Backend> = None;

        #[cfg(dht)]
        let dht = config
            .dht
            .map(|dht| Backend::dht(dht, config.dht_report_policy))
            .transpose()?;
        #[cfg(not(dht))]
        let dht: Option<Backend> = None;

        let backend = match (dht, relay) {
            (Some(dht), None) => dht,
            (None, Some(relay)) => relay,
            (None, None) => return Err(BuildError::NoNetwork),
            (Some(dht), Some(relay)) => dht
                .checked_combine(relay)
                .expect("failed to merge dht and relays backends"),
        };

        Ok(Self {
            minimum_ttl: config.minimum_ttl,
            maximum_ttl: config.maximum_ttl,
            cache,
            cache_write_lock: Arc::new(Mutex::new(())),
            backend: Arc::new(backend),
            #[cfg(feature = "endpoints")]
            max_recursion_depth: config.max_recursion_depth,
        })
    }

    /// Publishes a signed packet to the configured backend and updates the local cache.
    ///
    /// The client does not serialize concurrent publishes for the same public key. If
    /// multiple tasks may publish different packets for one key, coordinate those
    /// publishes in application code.
    ///
    /// # Returns
    ///
    /// Returns a [`StoredNodeCount`] with the number of DHT nodes that
    /// acknowledged storing the packet. When multiple publishing backends are
    /// configured, this is the maximum count reported by any successful
    /// backend, not a sum, because multiple backends may store the packet on
    /// the same DHT nodes. When publishing only through relays, older relays
    /// that do not return this count are treated as if one DHT node
    /// acknowledged storing the packet.
    ///
    /// # Errors
    ///
    /// Returns an error when publishing fails on the configured backend, or when the
    /// local cache already contains a more recent packet for the same public key.
    pub async fn publish(&self, packet: &SignedPacket) -> Result<StoredNodeCount, PublishError> {
        async_compat_if_necessary(self.publish_inner(packet)).await
    }

    async fn publish_inner(&self, packet: &SignedPacket) -> Result<StoredNodeCount, PublishError> {
        if let Some(cached) = self.get_cached(&packet.public_key()) {
            validate_cached_publish(packet, &cached)?;
        }

        let stored_on = self.backend.publish(packet).await?;
        self.update_cache_if_needed(packet);

        Ok(stored_on)
    }

    /// Resolves a signed packet for `key` according to [`ResolvePolicy`].
    ///
    /// With [`ResolvePolicy::CacheFirst`], a fresh cached packet is returned
    /// immediately. An expired cached packet becomes a freshness floor while
    /// network responses are aggregated, so responses below that floor or
    /// expired under the client's TTL bounds do not stop pending resolution
    /// attempts. If no fresh response is found, the newest expired response may
    /// still update the cache before resolution returns [`ResolveError::NotFound`].
    /// Successful network results for every policy update the local cache
    /// without replacing a newer cached packet.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] when no acceptable packet is found or the
    /// configured backends fail to resolve one.
    pub async fn resolve(
        &self,
        key: &PublicKey,
        policy: ResolvePolicy,
    ) -> Result<SignedPacket, ResolveError> {
        async_compat_if_necessary(self.resolve_inner(key, policy)).await
    }

    async fn resolve_inner(
        &self,
        key: &PublicKey,
        policy: ResolvePolicy,
    ) -> Result<SignedPacket, ResolveError> {
        let packet = match policy {
            ResolvePolicy::CacheOnly => self.resolve_cache_only(key).await,
            ResolvePolicy::CacheFirst => self.resolve_cache_first(key).await,
            ResolvePolicy::NetworkOnly => {
                self.backend
                    .resolve(key, BackendResolvePolicy::NetworkOnly)
                    .await
            }
        }?;

        self.update_cache_if_needed(&packet);
        Ok(packet)
    }

    async fn resolve_cache_only(&self, key: &PublicKey) -> Result<SignedPacket, ResolveError> {
        if let Some(packet) = self.get_cached(key) {
            return Ok(packet);
        }
        self.backend
            .resolve(key, BackendResolvePolicy::CacheOnly)
            .await
    }

    async fn resolve_cache_first(&self, key: &PublicKey) -> Result<SignedPacket, ResolveError> {
        let cached = self.get_cached(key);
        if let Some(packet) = cached.as_ref() {
            if !packet.is_expired(self.minimum_ttl, self.maximum_ttl) {
                return Ok(packet.clone());
            }
        }

        let cache_context = CacheContext::new(cached.as_ref(), self.minimum_ttl, self.maximum_ttl);
        let packet = self
            .backend
            .resolve(key, BackendResolvePolicy::CacheFirst(cache_context))
            .await?;

        if cache_context.accepts_network_packet(&packet) {
            Ok(packet)
        } else {
            // The new packet can be expired but newer than the cached packet.
            self.update_cache_if_needed(&packet);
            Err(ResolveError::NotFound)
        }
    }

    fn get_cached(&self, key: &PublicKey) -> Option<SignedPacket> {
        let cache = self.cache.as_ref()?;
        let key = CacheKey::from(key);
        cache.get(&key)
    }

    fn update_cache_if_needed(&self, packet: &SignedPacket) {
        let Some(cache) = self.cache.as_ref() else {
            return;
        };

        let key = CacheKey::from(packet.public_key());

        // Keep the comparison and write atomic across this client's clones.
        // Other clients and direct cache writers need separate coordination;
        // see ClientBuilder::cache.
        let _lock = self
            .cache_write_lock
            .lock()
            .expect("Client cache_write_lock");

        let should_update = match cache.get_read_only(&key) {
            None => true,
            Some(cached) if packet.more_recent_than(&cached) => true,
            Some(cached) if packet.is_same_as(&cached) => {
                // The packet is the same but we refresh the last seen.
                packet.last_seen() > cached.last_seen()
            }
            _ => false,
        };

        if should_update {
            cache.put(&key, packet);
        }
    }

    /// Returns a reference to the internal cache.
    #[cfg(test)]
    pub(crate) fn cache(&self) -> Option<&dyn Cache> {
        self.cache.as_deref()
    }
}

fn validate_cached_publish(
    packet: &SignedPacket,
    cached: &SignedPacket,
) -> Result<(), PublishError> {
    if cached.more_recent_than(packet) {
        return Err(PublishError::NotMostRecent);
    }

    Ok(())
}

fn build_cache(cache_size: usize, cache: Option<Arc<dyn Cache>>) -> Option<Arc<dyn Cache>> {
    // A zero configured size or custom-cache capacity disables caching.
    let cache_size = NonZeroUsize::new(cache_size)?;

    match cache {
        Some(cache) if cache.capacity() > 0 => Some(cache),
        Some(_) => None,
        None => Some(Arc::new(InMemoryCache::new(cache_size))),
    }
}

async fn async_compat_if_necessary<O>(fut: impl Future<Output = O>) -> O {
    #[cfg(not(wasm_browser))]
    {
        if tokio::runtime::Handle::try_current().is_err() {
            return async_compat::Compat::new(fut).await;
        }
    }

    fut.await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct ReadableZeroCapacityCache(SignedPacket);

    impl Cache for ReadableZeroCapacityCache {
        fn len(&self) -> usize {
            1
        }

        fn put(&self, _key: &CacheKey, _signed_packet: &SignedPacket) {}

        fn get(&self, _key: &CacheKey) -> Option<SignedPacket> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn build_cache_disables_readable_custom_cache_with_zero_capacity() {
        let packet = SignedPacket::builder()
            .sign(&crate::Keypair::random())
            .unwrap();
        let key = packet.public_key().into();
        let cache: Arc<dyn Cache> = Arc::new(ReadableZeroCapacityCache(packet));

        assert!(cache.get(&key).is_some());
        assert!(build_cache(crate::DEFAULT_CACHE_SIZE, Some(cache)).is_none());
    }

    #[cfg(all(relays, not(wasm_browser)))]
    mod concurrent_updates {
        use super::*;
        use std::sync::mpsc;
        use std::time::Duration;

        #[derive(Debug)]
        struct ReadGate {
            started: mpsc::Sender<()>,
            resume: mpsc::Receiver<()>,
        }

        #[derive(Clone, Debug)]
        struct PausedReadCache {
            inner: InMemoryCache,
            first_read: Arc<Mutex<Option<ReadGate>>>,
        }

        impl Cache for PausedReadCache {
            fn capacity(&self) -> usize {
                self.inner.capacity()
            }

            fn len(&self) -> usize {
                self.inner.len()
            }

            fn put(&self, key: &CacheKey, packet: &SignedPacket) {
                self.inner.put(key, packet);
            }

            fn get(&self, key: &CacheKey) -> Option<SignedPacket> {
                self.inner.get(key)
            }

            fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
                let packet = self.inner.get_read_only(key);
                let gate = self.first_read.lock().unwrap().take();
                if let Some(gate) = gate {
                    // Pause after reading so a competing update can try to write
                    // before this caller acts on its cached snapshot.
                    gate.started.send(()).unwrap();
                    gate.resume.recv_timeout(Duration::from_secs(5)).unwrap();
                }
                packet
            }
        }

        #[rstest::rstest]
        #[case::newer_packet(2)]
        #[case::later_last_seen(1)]
        fn cloned_clients_preserve_freshest_cache_entry(#[case] newer_timestamp: u64) {
            let keypair = crate::Keypair::random();
            let mut older = SignedPacket::builder()
                .timestamp(1.into())
                .sign(&keypair)
                .unwrap();
            older.set_last_seen(&1.into());
            let mut newer = SignedPacket::builder()
                .timestamp(newer_timestamp.into())
                .sign(&keypair)
                .unwrap();
            newer.set_last_seen(&2.into());

            let (read_started, read_started_rx) = mpsc::channel();
            let (resume_read, resume_read_rx) = mpsc::channel();
            let cache = Arc::new(PausedReadCache {
                inner: InMemoryCache::new(NonZeroUsize::new(1).unwrap()),
                first_read: Arc::new(Mutex::new(Some(ReadGate {
                    started: read_started,
                    resume: resume_read_rx,
                }))),
            });
            let client = Client::builder()
                .no_default_network()
                .relays(&["http://localhost:6881"])
                .unwrap()
                .cache(cache.clone())
                .build()
                .unwrap();

            std::thread::scope(|scope| {
                let cloned_client = client.clone();
                let older_update = scope.spawn(move || {
                    cloned_client.update_cache_if_needed(&older);
                });
                read_started_rx
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();

                let (updated, updated_rx) = mpsc::channel();
                let client = &client;
                let newer = &newer;
                let newer_update = scope.spawn(move || {
                    client.update_cache_if_needed(newer);
                    updated.send(()).unwrap();
                });

                // Without the shared lock, the newer write can finish while
                // the older update is paused, then be overwritten on resume.
                // With the lock, the newer update waits until we resume the read.
                let _ = updated_rx.recv_timeout(Duration::from_millis(100));
                resume_read.send(()).unwrap();
                older_update.join().unwrap();
                newer_update.join().unwrap();
            });

            let cached = cache.get(&keypair.public_key().into()).unwrap();
            assert_eq!(cached.as_bytes(), newer.as_bytes());
            assert_eq!(cached.last_seen(), newer.last_seen());
        }
    }
}
