#[cfg(relays)]
use ntimestamp::Timestamp;

#[cfg(relays)]
use crate::ResolvePolicy;

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

#[cfg(relays)]
impl<'a> BackendResolvePolicy<'a> {
    /// Returns the cache context when this is a cache-first resolve.
    pub(super) const fn cache_context(self) -> Option<CacheContext<'a>> {
        match self {
            Self::CacheFirst(context) => Some(context),
            _ => None,
        }
    }

    /// Returns the corresponding relay resolve policy.
    pub(super) const fn as_relay_policy(self) -> ResolvePolicy {
        match self {
            Self::LocalOrRelayCacheOnly => ResolvePolicy::LocalOrRelayCacheOnly,
            Self::CacheFirst(_) => ResolvePolicy::CacheFirst,
            Self::DhtNetworkOnly => ResolvePolicy::DhtNetworkOnly,
        }
    }

    /// Returns the conditional request bound for relay resolution.
    pub(super) fn relay_request_lower_bound(self) -> Option<Timestamp> {
        self.cache_context()
            .and_then(CacheContext::relay_request_lower_bound)
    }

    /// Returns whether relay resolution completes on its first acceptable packet.
    pub(super) const fn completes_on_first_acceptable(self) -> bool {
        !matches!(self, Self::DhtNetworkOnly)
    }
}
