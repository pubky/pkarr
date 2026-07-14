use super::CacheContext;

/// Backend-specific resolution policy with all cache-first state grouped into
/// one variant.
#[derive(Clone, Copy, Debug)]
pub(in crate::client) enum BackendResolvePolicy<'a> {
    /// Resolve only from local or relay caches.
    LocalOrRelayCacheOnly,
    /// Apply a cached packet floor and local TTL bounds to network responses.
    CacheFirst(CacheContext<'a>),
    /// Query configured backends for DHT network state without cached packet state.
    DhtNetworkOnly,
}
