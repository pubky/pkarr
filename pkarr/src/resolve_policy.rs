//! Resolve policies.

use serde::Deserialize;

/// Controls whether resolution may use only cached packets,
/// prefer cached packets before querying the DHT, or bypass the cache and query
/// the DHT network directly.
///
/// | Policy | Cache behavior | DHT behavior | Typical use |
/// | --- | --- | --- | --- |
/// | [`ResolvePolicy::LocalOrRelayCacheOnly`] | Returns a cached packet, even if expired | Never queried | Fast local reads and republishing |
/// | [`ResolvePolicy::CacheFirst`] | Returns a cached packet only while it is fresh | Queries the first responder on cache miss or expiry | Normal application resolution while respecting TTLs |
/// | [`ResolvePolicy::DhtNetworkOnly`] | Ignores cached packets for the lookup, but updates the cache after retrieval | Queries all relevant DHT nodes for the newest packet. Slow but accurate. | When you need the absolute most recent packet, for example for recovering after stale sequence errors |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum ResolvePolicy {
    /// Return only a locally cached packet, even if expired.
    ///
    /// This is guaranteed to be fast and does not touch DHT nodes.
    LocalOrRelayCacheOnly,

    /// Return a cached packet if it is not expired, and request a new packet
    /// from the first responding DHT node otherwise.
    ///
    /// This is guaranteed to return only non-expired packets.
    CacheFirst,

    /// Query all relevant DHT nodes and update the cache.
    ///
    /// This is guaranteed to return the newest packet.
    DhtNetworkOnly,
}
