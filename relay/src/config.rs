//! Configuration for Pkarr relay

use anyhow::{Context, Result};
use pkarr::{DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::rate_limiting::RateLimiterConfig;

pub const DEFAULT_CACHE_SIZE: usize = 1_000_000;
pub const DEFAULT_HTTP_PORT: u16 = 6881;
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
    #[serde(default = "default_http_port")]
    pub port: u16,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            port: default_http_port(),
        }
    }
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct MainlineConfig {
    pub port: Option<u16>,
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
        validate_cache_ttl(&config.cache)?;

        Ok(config)
    }
}

fn default_http_port() -> u16 {
    DEFAULT_HTTP_PORT
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
    use super::{validate_cache_ttl, CacheConfig, RelayConfig};

    #[test]
    fn default_config_enables_rate_limiter() {
        assert!(RelayConfig::default().rate_limiter.is_some());
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
