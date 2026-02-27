//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

use crate::rate_limiter::OperationLimit;
use crate::rate_limiting::RateLimiterConfig;

/// Default cache size for the relay
pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
/// Directory name for cache storage
pub const CACHE_DIR: &str = "pkarr-cache";

#[derive(Serialize, Deserialize, Default)]
struct ConfigToml {
    http: Option<HttpConfig>,
    mainline: Option<MainlineConfig>,
    // Backward compatibility: [rate_limiter] applies to both DHT and HTTP
    rate_limiter: Option<RateLimiterConfig>,
    // New specific configs (override the global rate_limiter)
    dht_rate_limit: Option<RateLimiterConfig>,
    http_rate_limit: Option<HttpRateLimitConfig>,
    relay: Option<RelayConfig>,
    cache: Option<CacheConfig>,
}

#[derive(Serialize, Deserialize, Default)]
struct HttpRateLimitConfig {
    rate_limits: Option<Vec<OperationLimit>>,
}

#[derive(Serialize, Deserialize, Default)]
struct RelayConfig {
    behind_proxy: Option<bool>,
    resolve_most_recent: Option<bool>,
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
    /// DHT rate limiter configuration (IP-based, protects the DHT node)
    pub dht_rate_limiter: Option<RateLimiterConfig>,
    /// HTTP rate limiter configuration (operation-based, protects the HTTP server)
    pub http_rate_limiter: Option<Vec<OperationLimit>>,
    /// Whether this relay is behind a proxy (affects IP extraction for all rate limiting)
    ///
    /// If true, trusts X-Forwarded-For and X-Real-IP headers.
    /// If false, only uses the direct TCP connection IP.
    ///
    /// Defaults to false for security.
    pub behind_proxy: bool,
    /// Whether to respect the most_recent query parameter in resolve requests
    ///
    /// If true, the most_recent query parameter is respected and will bypass cache.
    /// If false, the most_recent query parameter is ignored and will always load from cache.
    ///
    /// Defaults to false.
    pub resolve_most_recent: bool,
}

impl Default for Config {
    fn default() -> Self {
        let mut this = Self {
            http_port: 6881,
            pkarr: Default::default(),
            cache_path: None,
            cache_size: DEFAULT_CACHE_SIZE,
            dht_rate_limiter: Some(RateLimiterConfig::default()),
            http_rate_limiter: None,
            behind_proxy: false,
            resolve_most_recent: false,
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

        // Backward compatibility: [rate_limiter] applies to both DHT and HTTP
        if let Some(rate_limiter) = config_toml.rate_limiter {
            config.dht_rate_limiter = Some(rate_limiter.clone());
            config.behind_proxy = rate_limiter.behind_proxy;
        }

        // New specific configs override the global rate_limiter
        config.dht_rate_limiter = config_toml.dht_rate_limit.or(config.dht_rate_limiter);

        // Apply HTTP rate limiter config
        if let Some(http_rate_limit) = config_toml.http_rate_limit {
            config.http_rate_limiter = http_rate_limit.rate_limits.or(config.http_rate_limiter);
        }

        // Apply relay configuration
        if let Some(relay_config) = config_toml.relay {
            if let Some(behind_proxy) = relay_config.behind_proxy {
                config.behind_proxy = behind_proxy;
            }
            config.resolve_most_recent = relay_config
                .resolve_most_recent
                .unwrap_or(config.resolve_most_recent);
        }

        Ok(config)
    }
}
