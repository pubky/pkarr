//! Resolve policies.

use serde::Deserialize;

/// Controls whether resolution may use only cached packets,
/// prefer cached packets before querying the DHT, or bypass cached reads and
/// query the DHT network directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum ResolvePolicy {
    /// Return only a locally cached or relay-cached packet, even if expired.
    ///
    /// This is guaranteed to be fast and does not touch DHT nodes.
    /// Useful for republishing.
    LocalOrRelayCacheOnly,

    /// Use the cache (local or relay) if the packet is available and is within the TTL.
    /// Query the DHT nodes on a cache miss. Use the first DHT response to
    /// guarantee a fast response.
    /// May return an outdated packet in edge cases.
    ///
    /// This is guaranteed to return only packets that are not expired.
    /// Useful for normal application resolution while respecting TTLs.
    CacheFirst,

    /// Query all relevant DHT nodes for the most recent packet observed and
    /// update the cache after retrieval.
    /// This is slower, but more accurate.
    ///
    /// This returns the most recent packet found during the query, but does not
    /// guarantee global consistency.
    /// Useful when you need the most recent packet, for example for recovering
    /// after stale sequence errors.
    DhtNetworkOnly,
}
