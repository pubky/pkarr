//! Configuration for the server

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

/// Server configuration
///
/// The config is usually loaded from a file with [`Self::load`].
///
/// The struct also implements [`Default`] which creates a config suitable for local development
/// and testing.
#[derive(Serialize, Deserialize)]
pub struct Config {
    /// TCP port to run the HTTP server on
    ///
    /// Defaults to `6881`
    relay_port: u16,
    /// UDP port to run the DHT on
    ///
    /// Defaults to `6881`
    dht_port: u16,
    /// Path to cache database
    ///
    /// Defaults to a directory in the OS data directory
    cache_path: Option<String>,
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
        self.relay_port
    }

    pub fn dht_port(&self) -> u16 {
        self.dht_port
    }

    /// Get the path to the cache database file.
    pub fn pkarr_cache_path(&self) -> Result<PathBuf> {
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

impl Default for Config {
    fn default() -> Self {
        Self {
            relay_port: 6881,
            dht_port: 6881,
            cache_path: None,
        }
    }
}

impl Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entry(&"relay_port", &self.relay_port)
            .entry(&"dht_port", &self.dht_port)
            .entry(&"cache_path", &self.pkarr_cache_path())
            .finish()
    }
}
