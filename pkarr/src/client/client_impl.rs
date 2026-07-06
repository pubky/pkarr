use ntimestamp::Timestamp;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::{
    Cache, CacheKey, ClientBuilder, InMemoryCache, PublicKey, ResolvePolicy, SignedPacket,
};

use super::backend::Backend;
use super::builder::Config;
use super::errors::ResolveError;
use super::{BuildError, ConcurrencyError, PublishError};

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
    /// Returns a builder to edit config before creating Client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub(crate) fn new(config: Config) -> Result<Self, BuildError> {
        let cache = build_cache(config.cache_size, config.cache);

        #[cfg(relays)]
        let relays = config
            .relays
            .map(|relays| Backend::relays(relays, config.request_timeout, config.reqwest_client))
            .transpose()?;
        #[cfg(not(relays))]
        let relays: Option<Backend> = None;

        #[cfg(dht)]
        let dht = config
            .dht
            .map(|dht| Backend::dht(dht, config.dht_report_policy))
            .transpose()?;
        #[cfg(not(dht))]
        let dht: Option<Backend> = None;

        let backend = match (dht, relays) {
            (Some(dht), None) => dht,
            (None, Some(relays)) => relays,
            (None, None) => return Err(BuildError::NoNetwork),
            (Some(dht), Some(relays)) => dht
                .try_merge(relays)
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
    /// publishes in application code and use `cas` to detect network-side conflicts.
    ///
    /// # Returns
    ///
    /// Returns `stored_on`, the number of DHT nodes that acknowledged storing
    /// the packet. When multiple publishing backends are configured, this is
    /// the maximum count reported by any successful backend, not a sum, because
    /// multiple backends may store the packet on the same DHT nodes. When
    /// publishing only through relays, older relays that do not return this
    /// count are treated as if one DHT node acknowledged storing the packet.
    ///
    /// # Errors
    ///
    /// Returns an error when publishing fails on the configured backend, or when the
    /// local cache already contains a more recent packet for the same public key.
    pub async fn publish(
        &self,
        packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<u32, PublishError> {
        async_compat_if_necessary(self.publish_inner(packet, cas)).await
    }

    async fn publish_inner(
        &self,
        packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<u32, PublishError> {
        if let Some(cached) = self.get_cached(&packet.public_key()) {
            validate_cached_publish(packet, &cached, cas)?;
        }

        let stored_on = self.backend.publish(packet, cas).await?;
        self.update_cache_if_needed(packet);

        Ok(stored_on)
    }

    /// Resolves a signed packet for `key` according to `policy`.
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
            ResolvePolicy::LocalOrRelayCacheOnly => self.resolve_cache_only(key).await,
            ResolvePolicy::CacheFirst => self.resolve_cache_first(key).await,
            ResolvePolicy::DhtNetworkOnly => {
                self.backend
                    .resolve(key, ResolvePolicy::DhtNetworkOnly, None)
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
            .resolve(key, ResolvePolicy::LocalOrRelayCacheOnly, None)
            .await
    }

    async fn resolve_cache_first(&self, key: &PublicKey) -> Result<SignedPacket, ResolveError> {
        let cached = self.get_cached(key);

        if let Some(ref packet) = cached {
            if !packet.is_expired(self.minimum_ttl, self.maximum_ttl) {
                return Ok(packet.clone());
            }
        }

        let more_recent_than = cached
            .as_ref()
            .map(SignedPacket::timestamp)
            .and_then(decrement);
        let packet = self
            .backend
            .resolve(key, ResolvePolicy::CacheFirst, more_recent_than)
            .await;

        match cached {
            Some(cached) => floor_cache_result(packet, cached),
            None => packet,
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
    cas: Option<Timestamp>,
) -> Result<(), PublishError> {
    if cached.more_recent_than(packet) {
        return Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent));
    }

    if let Some(cas) = cas {
        if cached.timestamp().as_u64() > cas.as_u64() {
            return Err(PublishError::Concurrency(ConcurrencyError::CasFailed));
        }
    }

    Ok(())
}

fn build_cache(cache_size: usize, cache: Option<Arc<dyn Cache>>) -> Option<Arc<dyn Cache>> {
    // A zero cache size disables caching, including any provided cache.
    let cache_size = NonZeroUsize::new(cache_size)?;
    cache.or_else(|| Some(Arc::new(InMemoryCache::new(cache_size))))
}

fn decrement(timestamp: Timestamp) -> Option<Timestamp> {
    timestamp.as_u64().checked_sub(1).map(Timestamp::from)
}

/// Applies the cached packet as the freshness floor for a CacheFirst network result.
fn floor_cache_result(
    result: Result<SignedPacket, ResolveError>,
    cached: SignedPacket,
) -> Result<SignedPacket, ResolveError> {
    match result {
        Ok(packet) if cached.more_recent_than(&packet) => Err(ResolveError::NotFound),
        Ok(packet) => Ok(packet),
        Err(ResolveError::InvalidSignedPacket { seq })
            if cached.timestamp().as_u64() as i64 >= seq =>
        {
            Err(ResolveError::NotFound)
        }
        Err(e) => Err(e),
    }
}

async fn async_compat_if_necessary<T, O>(fut: T) -> O
where
    T: Future<Output = O>,
{
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
    use crate::Keypair;

    #[test]
    fn floor_cache_result_treats_equal_invalid_seq_as_covered_by_cache() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();

        assert_eq!(
            floor_cache_result(Err(ResolveError::InvalidSignedPacket { seq: 10 }), cached),
            Err(ResolveError::NotFound)
        );
    }
}
