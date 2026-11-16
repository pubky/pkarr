//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use ipnet::IpNet;
use pkarr::{DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};
use serde::{Deserialize, Serialize};
use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
};
use tokio::sync::Semaphore;

use crate::rate_limiting::RateLimiterConfig;

pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
pub const DEFAULT_HTTP_PORT: u16 = 6881;
pub const DEFAULT_HTTP_MAX_CONNECTIONS: usize = 1_024;
pub const DEFAULT_HTTP_MAX_CONNECTION_AGE_SECONDS: u64 = 5 * 60;
pub const CACHE_DIR: &str = "pkarr-cache";

#[derive(Serialize, Deserialize, Debug)]
pub struct RelayConfig {
    #[serde(default)]
    pub http: HttpConfig,
    #[serde(default)]
    pub mainline: MainlineConfig,
    pub rate_limiter: Option<RateLimiterConfig>,
    #[serde(default)]
    pub cache: CacheConfig,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            http: HttpConfig::default(),
            mainline: MainlineConfig::default(),
            rate_limiter: Some(RateLimiterConfig::default()),
            cache: CacheConfig::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HttpConfig {
    #[serde(default)]
    pub client_identity: ClientIdentityConfig,
    #[serde(default = "default_http_port")]
    pub port: u16,
    #[serde(default = "default_http_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_http_max_connection_age_seconds")]
    pub max_connection_age_seconds: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            client_identity: ClientIdentityConfig::default(),
            port: default_http_port(),
            max_connections: default_http_max_connections(),
            max_connection_age_seconds: default_http_max_connection_age_seconds(),
        }
    }
}

/// Determines which client IP is used for HTTP and user DHT rate limits.
///
/// Additional identity modes may be added in future releases.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClientIdentityConfig {
    /// Use the TCP peer address and ignore forwarding headers (the default).
    // Keep this a struct variant so Serde rejects fields for header mode.
    Direct {},
    /// Require a single-IP header on every request from a trusted proxy.
    HttpHeader {
        /// Header the proxy overwrites with a bare IPv4 or IPv6 address.
        header: String,
        /// Immediate TCP peer networks allowed to supply client addresses.
        ///
        /// Must be nonempty. Other peers are disconnected before HTTP parsing.
        trusted_proxies: Vec<IpNet>,
    },
}

impl Default for ClientIdentityConfig {
    fn default() -> Self {
        Self::Direct {}
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MainlineConfig {
    pub port: Option<u16>,
    pub public_ip: Option<Ipv4Addr>,
    /// Maximum duration of an individual DHT request, in seconds.
    #[serde(default = "default_mainline_request_timeout_seconds")]
    pub request_timeout_seconds: u64,
}

impl Default for MainlineConfig {
    fn default() -> Self {
        Self {
            port: None,
            public_ip: None,
            request_timeout_seconds: default_mainline_request_timeout_seconds(),
        }
    }
}

fn default_mainline_request_timeout_seconds() -> u64 {
    2
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CacheConfig {
    pub path: Option<PathBuf>,
    #[serde(default = "default_cache_size")]
    pub size: usize,
    #[serde(default = "default_minimum_ttl")]
    pub minimum_ttl: u32,
    #[serde(default = "default_maximum_ttl")]
    pub maximum_ttl: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            path: None,
            size: default_cache_size(),
            minimum_ttl: default_minimum_ttl(),
            maximum_ttl: default_maximum_ttl(),
        }
    }
}

impl RelayConfig {
    /// Load the config from a file.
    pub async fn load(path: impl AsRef<Path>) -> Result<RelayConfig> {
        let content = tokio::fs::read_to_string(path.as_ref())
            .await
            .with_context(|| format!("failed to read {}", path.as_ref().to_string_lossy()))?;

        let mut config: RelayConfig = toml::from_str(&content)?;
        config.cache.path = config
            .cache
            .path
            .map(|cache_path| resolve_cache_path(cache_path, path.as_ref()))
            .transpose()?;
        validate_http(&config.http)?;
        validate_cache_ttl(&config.cache)?;

        Ok(config)
    }
}

fn default_http_port() -> u16 {
    DEFAULT_HTTP_PORT
}

fn default_http_max_connections() -> usize {
    DEFAULT_HTTP_MAX_CONNECTIONS
}

fn default_http_max_connection_age_seconds() -> u64 {
    DEFAULT_HTTP_MAX_CONNECTION_AGE_SECONDS
}

fn default_cache_size() -> usize {
    DEFAULT_CACHE_SIZE
}

fn default_minimum_ttl() -> u32 {
    DEFAULT_MINIMUM_TTL
}

fn default_maximum_ttl() -> u32 {
    DEFAULT_MAXIMUM_TTL
}

fn validate_http(http: &HttpConfig) -> Result<()> {
    anyhow::ensure!(
        http.max_connections > 0,
        "http.max_connections must be greater than zero"
    );
    anyhow::ensure!(
        http.max_connections <= Semaphore::MAX_PERMITS,
        "http.max_connections must be less than or equal to {}",
        Semaphore::MAX_PERMITS
    );
    anyhow::ensure!(
        http.max_connection_age_seconds > 0,
        "http.max_connection_age_seconds must be greater than zero"
    );

    Ok(())
}

fn validate_cache_ttl(cache: &CacheConfig) -> Result<()> {
    anyhow::ensure!(
        cache.minimum_ttl <= cache.maximum_ttl,
        "cache.minimum_ttl must be less than or equal to cache.maximum_ttl"
    );

    Ok(())
}

fn resolve_cache_path(cache_path: PathBuf, config_file_path: &Path) -> Result<PathBuf> {
    if cache_path.is_absolute() {
        return Ok(cache_path);
    }

    let config_dir = config_file_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", config_file_path.display()))?
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    config_dir
        .join(&cache_path)
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", cache_path.display()))
}

#[cfg(test)]
mod tests {
    use super::{
        validate_cache_ttl, validate_http, CacheConfig, HttpConfig, RelayConfig,
        DEFAULT_HTTP_MAX_CONNECTIONS, DEFAULT_HTTP_MAX_CONNECTION_AGE_SECONDS,
    };

    #[test]
    fn client_identity_defaults_to_direct_and_round_trips() {
        use super::ClientIdentityConfig;
        for input in ["", "[http]\nport = 1234"] {
            let config: RelayConfig = toml::from_str(input).unwrap();
            assert_eq!(config.http.client_identity, ClientIdentityConfig::Direct {});
        }
        let config: RelayConfig = toml::from_str(
            r#"[http.client_identity]
mode = "http_header"
header = "X-Real-IP"
trusted_proxies = ["127.0.0.1/32", "::1/128"]"#,
        )
        .unwrap();
        validate_http(&config.http).unwrap();
        let encoded = toml::to_string(&config.http.client_identity).unwrap();
        assert_eq!(
            toml::from_str::<ClientIdentityConfig>(&encoded).unwrap(),
            config.http.client_identity
        );
    }

    #[test]
    fn invalid_client_identity_schema_is_rejected() {
        for identity in [
            "mode = 'http_header'\nheader = 'X-Real-IP'",
            "mode = 'http_header'\ntrusted_proxies = ['127.0.0.1/32']",
            "mode = 'http_header'\nheader = 'X-Real-IP'\ntrusted_proxies = ['invalid']",
            "mode = 'direct'\nheader = 'X-Real-IP'",
            "mode = 'direct'\ntrusted_proxies = ['127.0.0.1/32']",
            "mode = 'proxy_protocol'",
            "mode = 'http_header'\nheader = 'X-Real-IP'\ntrusted_proxies = ['127.0.0.1/32']\nextra = true",
        ] {
            let parsed = toml::from_str::<RelayConfig>(&format!("[http.client_identity]\n{identity}"));
            assert!(parsed.is_err(), "{identity}");
        }
    }

    #[test]
    fn default_config_enables_rate_limiter() {
        assert!(RelayConfig::default().rate_limiter.is_some());
    }

    #[test]
    fn http_connection_limits_default_when_omitted() {
        let config: RelayConfig = toml::from_str("[http]\nport = 1234").unwrap();

        assert_eq!(config.http.max_connections, DEFAULT_HTTP_MAX_CONNECTIONS);
        assert_eq!(
            config.http.max_connection_age_seconds,
            DEFAULT_HTTP_MAX_CONNECTION_AGE_SECONDS
        );
    }

    #[test]
    fn http_connection_limits_can_be_overridden() {
        let config: RelayConfig =
            toml::from_str("[http]\nmax_connections = 2048\nmax_connection_age_seconds = 600")
                .unwrap();

        assert_eq!(config.http.max_connections, 2_048);
        assert_eq!(config.http.max_connection_age_seconds, 600);
    }

    #[test]
    fn http_connection_limits_must_be_nonzero() {
        let mut http = HttpConfig {
            max_connections: 0,
            ..Default::default()
        };
        assert!(validate_http(&http).is_err());

        http.max_connections = 1;
        http.max_connection_age_seconds = 0;
        assert!(validate_http(&http).is_err());
    }

    #[test]
    fn max_connections_must_not_exceed_semaphore_limit() {
        let http = HttpConfig {
            max_connections: tokio::sync::Semaphore::MAX_PERMITS + 1,
            ..Default::default()
        };

        let error = validate_http(&http).unwrap_err();

        assert!(error
            .to_string()
            .contains("http.max_connections must be less than or equal to"));
    }

    #[test]
    fn cache_ttl_bounds_must_be_ordered() {
        let cache = CacheConfig {
            minimum_ttl: 60,
            maximum_ttl: 30,
            ..Default::default()
        };

        let err = validate_cache_ttl(&cache).unwrap_err();

        assert!(err
            .to_string()
            .contains("cache.minimum_ttl must be less than or equal to cache.maximum_ttl"));
    }
}
