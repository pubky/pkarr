//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    num::NonZero,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::rate_limiter::{Operation, OperationLimit, QuotaValue};

/// Default cache size for the relay
pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
/// Directory name for cache storage
pub const CACHE_DIR: &str = "pkarr-cache";

#[derive(Serialize, Deserialize, Default)]
struct ConfigToml {
    http: Option<HttpConfig>,
    mainline: Option<MainlineConfig>,
    relay: Option<RelayToml>,
    cache: Option<CacheConfig>,
}

#[derive(Serialize, Deserialize, Default)]
struct RelayToml {
    rate_limits: Option<Vec<OperationLimit>>,
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
    /// Operation-based rate limiter configuration
    pub rate_limiter: Option<Vec<OperationLimit>>,
}

impl Default for Config {
    fn default() -> Self {
        // Default rate limiting configuration - same as historical relay settings
        let default_rate_limits = vec![
            OperationLimit {
                operation: Operation::Resolve,
                quota: QuotaValue::from_str("2r/s").expect("valid default quota"),
                burst: Some(NonZero::new(10).expect("valid burst")),
                whitelist: vec![],
            },
            OperationLimit {
                operation: Operation::ResolveMostRecent,
                quota: QuotaValue::from_str("2r/s").expect("valid default quota"),
                burst: Some(NonZero::new(10).expect("valid burst")),
                whitelist: vec![],
            },
            OperationLimit {
                operation: Operation::Publish,
                quota: QuotaValue::from_str("2r/s").expect("valid default quota"),
                burst: Some(NonZero::new(10).expect("valid burst")),
                whitelist: vec![],
            },
        ];

        let mut this = Self {
            http_port: 6881,
            pkarr: Default::default(),
            cache_path: None,
            cache_size: DEFAULT_CACHE_SIZE,
            rate_limiter: Some(default_rate_limits),
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

        // Use TOML-defined rate limits if present, otherwise keep defaults
        if let Some(rate_limits) = config_toml.relay.and_then(|r| r.rate_limits) {
            // Override defaults with TOML-defined rate limits
            config.rate_limiter = Some(rate_limits);
        }

        Ok(config)
    }
}
