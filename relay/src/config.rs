//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    net::IpAddr,
    path::{Path, PathBuf},
};

/// Relay operating modes that affect caching and resolution behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelayMode {
    /// Legacy mode uses cache and rate limiting, ignores most_recent query param.
    LEGACY,
    /// Public mode uses cache and rate limiting, respects most_recent query param.
    PUBLIC,
    /// Private mode uses cache and rate limiting, respects most_recent query param, but also skips rate limiting for whitelisted IPs.
    PRIVATE,
}

/// Default cache size for the relay
pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
/// Directory name for cache storage
pub const CACHE_DIR: &str = "pkarr-cache";

use crate::rate_limiting::RateLimiterConfig;

#[derive(Serialize, Deserialize, Default)]
struct ConfigToml {
    mode: Option<RelayMode>,
    http: Option<HttpConfig>,
    mainline: Option<MainlineConfig>,
    rate_limiter: Option<RateLimiterConfig>,
    cache: Option<CacheConfig>,
    ip_whitelist: Option<Vec<String>>,
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

    /// Relay operating mode
    /// Defaults to `PUBLIC`
    pub mode: RelayMode,
    /// IP whitelist configuration
    pub ip_whitelist: IpWhitelist,
}

#[derive(Debug, Clone, Default)]
pub struct IpWhitelist {
    /// Parsed IP addresses and CIDR blocks
    pub ips: Vec<ipnet::IpNet>,
}

impl IpWhitelist {
    /// Check if an IP address is trusted
    pub fn is_trusted(&self, ip: &IpAddr) -> bool {
        self.ips.iter().any(|net| net.contains(ip))
    }
}

impl Default for Config {
    fn default() -> Self {
        let mut this = Self {
            http_port: 6881,
            pkarr: Default::default(),
            cache_path: None,
            cache_size: DEFAULT_CACHE_SIZE,
            rate_limiter: Some(RateLimiterConfig::default()),
            mode: RelayMode::PUBLIC,
            ip_whitelist: IpWhitelist::default(),
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

        if let Some(mode) = config_toml.mode {
            config.mode = mode;
        }

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

        if let Some(ip_strings) = config_toml.ip_whitelist {
            let mut whitelist_ips = Vec::new();
            for ip_str in ip_strings {
                // Support both individual IPs and CIDR blocks
                match ip_str.parse::<ipnet::IpNet>() {
                    Ok(net) => whitelist_ips.push(net),
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Invalid whitelist IP/CIDR '{}': {}",
                            ip_str,
                            e
                        ));
                    }
                }
            }
            config.ip_whitelist = IpWhitelist { ips: whitelist_ips };
        }

        Ok(config)
    }
}
