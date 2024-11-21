//! Configuration for the server

use anyhow::{anyhow, Context, Result};
use pkarr::{DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

use crate::rate_limiting::RateLimiterConfig;

/// Server configuration
///
/// The config is usually loaded from a file with [`Self::load`].
#[derive(Serialize, Deserialize, Default)]
pub struct Config {
    /// TCP port to run the HTTP server on
    ///
    /// Defaults to `6881`
    relay_port: Option<u16>,
    /// UDP port to run the DHT on
    ///
    /// Defaults to `6881`
    dht_port: Option<u16>,
    /// Path to cache database
    ///
    /// Defaults to a directory in the OS data directory
    cache_path: Option<String>,
    /// See [pkarr::client::Settings::cache_size]
    cache_size: Option<usize>,
    /// Resolvers
    ///
    /// Other servers to query in parallel with the Dht queries
    ///
    /// See [pkarr::client::Settings::resolvers]
    resolvers: Option<Vec<String>>,
    /// See [pkarr::Settings::minimum_ttl]
    minimum_ttl: Option<u32>,
    /// See [pkarr::Settings::maximum_ttl]
    maximum_ttl: Option<u32>,
    rate_limiter: RateLimiterConfig,
}

impl Config {
    /// Load the config from a file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Config> {
        let s = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("failed to read {}", path.as_ref().to_string_lossy()))?;
        let config: Config = toml::from_str(&s)?;
        Ok(config)
    }

    pub fn relay_port(&self) -> u16 {
        self.relay_port.unwrap_or(6881)
    }

    pub fn dht_port(&self) -> u16 {
        self.dht_port.unwrap_or(6881)
    }

    pub fn resolvers(&self) -> Option<Vec<String>> {
        self.resolvers.clone()
    }

    pub fn minimum_ttl(&self) -> u32 {
        self.minimum_ttl.unwrap_or(DEFAULT_MINIMUM_TTL)
    }

    pub fn maximum_ttl(&self) -> u32 {
        self.maximum_ttl.unwrap_or(DEFAULT_MAXIMUM_TTL)
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size.unwrap_or(DEFAULT_CACHE_SIZE)
    }

    pub fn rate_limiter(&self) -> &RateLimiterConfig {
        &self.rate_limiter
    }

    /// Get the path to the cache database file.
    pub fn cache_path(&self) -> Result<PathBuf> {
        let dir = if let Some(cache_path) = &self.cache_path {
            PathBuf::from(cache_path)
        } else {
            let path = dirs_next::data_dir().ok_or_else(|| {
                anyhow!("operating environment provides no directory for application data")
            })?;
            path.join("pkarr-server")
        };

        Ok(dir.join("pkarr-cache"))
    }
}

impl Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entry(&"relay_port", &self.relay_port())
            .entry(&"dht_port", &self.dht_port())
            .entry(&"cache_path", &self.cache_path())
            .entry(&"cache_size", &self.cache_size())
            .entry(&"resolvers", &self.resolvers.clone().unwrap_or_default())
            .entry(&"minimum_ttl", &self.minimum_ttl())
            .entry(&"maximum_ttl", &self.maximum_ttl())
            .entry(&"rate_limter", &self.rate_limiter())
            .finish()
    }
}
