use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;

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
            .map(|relays| Backend::relay(relays, config.request_timeout, config.reqwest_client))
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

        // TODO: Make it atomic.
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
}
