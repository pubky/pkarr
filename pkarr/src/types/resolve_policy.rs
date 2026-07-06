//! Resolve policies.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Controls whether resolution may use only cached packets,
/// prefer cached packets before querying the DHT, or bypass cached reads and
/// query the DHT network directly.
///
/// Caches store valid `SignedPacket`s. Invalid DHT mutable items are reported
/// by sequence number, but are not stored in the packet cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvePolicy {
    /// Return only a locally cached or relay-cached packet, even if expired.
    ///
    /// This is guaranteed to be fast and does not touch DHT nodes.
    /// Useful for republishing.
    LocalOrRelayCacheOnly,

    /// Resolve with the fastest fresh answer without going backward.
    ///
    /// Returns a cached packet when it is still within TTL. Otherwise queries
    /// the DHT and returns the first acceptable result.
    ///
    /// If an expired cached packet exists, its sequence is used as a floor:
    /// older DHT packets, and invalid DHT mutable items at or below that
    /// sequence, are treated as stale rather than returned.
    ///
    /// This is the default policy for normal application resolution. It never
    /// returns expired packets, but it may miss a newer DHT value because it
    /// stops at the first acceptable response.
    CacheFirst,

    /// Query all relevant DHT nodes for the most recent value observed.
    /// This is slower, but more accurate.
    ///
    /// This policy ignores cached packets while querying and interpreting the
    /// DHT result. If the DHT currently contains an older valid packet than the
    /// cache, that older packet is returned.
    ///
    /// If the result contains a valid signed packet, the cache is updated only
    /// when that packet is newer than the cached packet, or is the same packet
    /// with a fresher last-seen timestamp. This policy never downgrades the
    /// cache to an older packet.
    ///
    /// This is guaranteed to return the newest valid signed packet, or the sequence
    /// number of a newer mutable item that is not a valid signed packet.
    /// A newer invalid signed packet is treated as the current DHT state.
    /// Useful when you need to account for the most recent DHT state, for
    /// example when recovering after stale sequence errors.
    DhtNetworkOnly,
}

impl ResolvePolicy {
    /// Return the canonical string representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalOrRelayCacheOnly => "LocalOrRelayCacheOnly",
            Self::CacheFirst => "CacheFirst",
            Self::DhtNetworkOnly => "DhtNetworkOnly",
        }
    }
}

impl fmt::Display for ResolvePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ResolvePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LocalOrRelayCacheOnly" => Ok(Self::LocalOrRelayCacheOnly),
            "CacheFirst" => Ok(Self::CacheFirst),
            "DhtNetworkOnly" => Ok(Self::DhtNetworkOnly),
            _ => Err(format!("invalid resolve policy: {s}")),
        }
    }
}

impl Serialize for ResolvePolicy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResolvePolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_parse_roundtrip() {
        for policy in [
            ResolvePolicy::LocalOrRelayCacheOnly,
            ResolvePolicy::CacheFirst,
            ResolvePolicy::DhtNetworkOnly,
        ] {
            let s = policy.to_string();
            assert_eq!(s, policy.as_str());
            assert_eq!(s.parse::<ResolvePolicy>(), Ok(policy));
        }
    }

    #[test]
    fn rejects_unknown_policy() {
        assert!("Unknown".parse::<ResolvePolicy>().is_err());
    }
}
