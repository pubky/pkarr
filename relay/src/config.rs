//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
pub const CACHE_DIR: &str = "pkarr-cache";

use crate::rate_limiting::RateLimiterConfig;

#[derive(Serialize, Deserialize, Default)]
struct ConfigToml {
    http: Option<HttpConfig>,
    mainline: Option<MainlineConfig>,
    rate_limiter: Option<RateLimiterConfig>,
    cache: Option<CacheConfig>,
}

#[derive(Serialize, Deserialize, Default)]
struct HttpConfig {
    port: Option<u16>,
}

#[derive(Serialize, Deserialize, Default)]
struct MainlineConfig {
    port: Option<u16>,
}

#[derive(Serialize, Deserialize, Default)]
struct CacheConfig {
    path: Option<PathBuf>,
    size: Option<usize>,
    /// See [pkarr::ClientBuilder::minimum_ttl]
    minimum_ttl: Option<u32>,
    /// See [pkarr::ClientBuilder::maximum_ttl]
    maximum_ttl: Option<u32>,
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
    /// Pkarr client builder
    pub pkarr: pkarr::ClientBuilder,
    /// Path to cache database
    ///
    /// Defaults to a directory in the OS data directory
    pub cache_path: Option<PathBuf>,
    /// See [pkarr::ClientBuilder::cache_size]
    ///
    /// Defaults to 1000_000
    pub cache_size: usize,
    /// IP rete limiter configuration
    pub rate_limiter: Option<RateLimiterConfig>,
}

impl Default for Config {
    fn default() -> Self {
        let mut this = Self {
            http_port: 6881,
            pkarr: Default::default(),
            cache_path: None,
            cache_size: DEFAULT_CACHE_SIZE,
            rate_limiter: Some(RateLimiterConfig::default()),
        };

        this.pkarr.no_relays();

        this
    }
}

impl Config {
    /// Load the config from a file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Config> {
        let config_file_path = path.as_ref();

        let s = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("failed to read {}", path.as_ref().to_string_lossy()))?;

        let config_toml: ConfigToml = toml::from_str(&s)?;

        let mut config = Config::default();

        if let Some(cache_config) = config_toml.cache {
            if let Some(ttl) = cache_config.minimum_ttl {
                config.pkarr.minimum_ttl(ttl);
            }
            if let Some(ttl) = cache_config.maximum_ttl {
                config.pkarr.maximum_ttl(ttl);
            }

            if let Some(cache_path) = cache_config.path.as_ref() {
                config.cache_path = Some(if cache_path.is_relative() {
                    // support relative path.
                    config_file_path
                        .canonicalize()
                        .expect("valid config path")
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join(cache_path)
                        .canonicalize()
                        .expect("valid cache path")
                } else {
                    cache_path.clone()
                });
            }

            config.cache_size = cache_config.size.unwrap_or(DEFAULT_CACHE_SIZE);
        }

        if let Some(port) = config_toml.mainline.and_then(|m| m.port) {
            config.pkarr.dht(|builder| builder.port(port));
        }

        if let Some(HttpConfig { port: Some(port) }) = config_toml.http {
            config.http_port = port;
        }

        config.rate_limiter = config_toml.rate_limiter;

        Ok(config)
    }
}
