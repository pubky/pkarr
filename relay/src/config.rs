//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;

use crate::rate_limiting::RateLimiterConfig;

#[derive(Serialize, Deserialize, Default)]
struct ConfigToml {
    http: Option<HttpConfig>,
    mainline: Option<MainlineConfig>,
    cache_path: Option<String>,
    cache_size: Option<usize>,
    /// See [pkarr::Config::minimum_ttl]
    minimum_ttl: Option<u32>,
    /// See [pkarr::Config::maximum_ttl]
    maximum_ttl: Option<u32>,
    rate_limiter: Option<RateLimiterConfig>,
}

#[derive(Serialize, Deserialize, Default)]
struct HttpConfig {
    port: Option<u16>,
}

#[derive(Serialize, Deserialize, Default)]
struct MainlineConfig {
    port: Option<u16>,
}

/// Pkarr Relay configuration
///
/// The config is usually loaded from a file with [`Self::load`].
#[derive(Debug)]
pub struct Config {
    /// TCP port to run the HTTP server on
    ///
    /// Defaults to `6881`
    pub http_port: u16,
    /// Pkarr client configuration
    pub pkarr_config: pkarr::Config,
    /// Path to cache database
    ///
    /// Defaults to a directory in the OS data directory
    pub cache_path: Option<PathBuf>,
    /// See [pkarr::client::Config::cache_size]
    ///
    /// Defaults to 1000_000
    pub cache_size: usize,
    /// IP rete limiter configuration
    pub rate_limiter: Option<RateLimiterConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            http_port: 6881,
            pkarr_config: pkarr::Config::default(),
            cache_path: None,
            cache_size: DEFAULT_CACHE_SIZE,
            rate_limiter: Some(RateLimiterConfig::default()),
        }
    }
}

impl Config {
    /// Load the config from a file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Config> {
        let s = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("failed to read {}", path.as_ref().to_string_lossy()))?;

        let config_toml: ConfigToml = toml::from_str(&s)?;

        let mut config = Config::default();

        if let Some(ttl) = config_toml.minimum_ttl {
            config.pkarr_config.minimum_ttl = ttl;
        }

        if let Some(ttl) = config_toml.maximum_ttl {
            config.pkarr_config.maximum_ttl = ttl;
        }

        config.pkarr_config.dht_config.port = config_toml.mainline.and_then(|m| m.port);

        if let Some(HttpConfig { port: Some(port) }) = config_toml.http {
            config.http_port = port;
        }

        if let Some(path) = config_toml.cache_path {
            config.cache_path = Some(PathBuf::from(path).join("pkarr-cache"));
        }

        config.cache_size = config_toml.cache_size.unwrap_or(DEFAULT_CACHE_SIZE);
        config.rate_limiter = config_toml.rate_limiter;

        Ok(config)
    }
}
