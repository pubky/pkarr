//! Resolve policies.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Controls whether resolution may use only cached packets,
/// prefer cached packets before querying the DHT, or bypass cached reads and
/// query the DHT network directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// This is guaranteed to return the newest packet.
    /// Useful when you need the most recent packet, for example for recovering
    /// after stale sequence errors.
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
