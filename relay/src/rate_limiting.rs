use std::{net::IpAddr, sync::Arc, time::Duration};

use crate::real_ip::RealIpKeyExtractor;
use axum::Router;
use governor::middleware::{NoOpMiddleware, StateInformationMiddleware};
use serde::{Deserialize, Deserializer, Serialize};
use tower_governor::governor::{GovernorConfig, GovernorConfigBuilder};
pub use tower_governor::GovernorLayer;

#[derive(Serialize, Debug)]
/// Configurations for rate limitng.
pub struct RateLimiterConfig {
    /// Enable rate limit based on headers commonly used by reverse proxies.
    ///
    /// Uses headers commonly used by reverse proxies to extract the original IP address,
    /// falling back to the connection's peer IP address.
    /// <https://docs.rs/tower_governor/latest/tower_governor/key_extractor/struct.SmartIpKeyExtractor.html>
    pub behind_proxy: bool,
    /// Set the interval after which one element of the quota is replenished in seconds.
    ///
    /// **The interval must not be zero.**
    pub per_second: u64,
    /// Set quota size that defines how many requests can occur
    /// before the governor middleware starts blocking requests from an IP address and
    /// clients have to wait until the elements of the quota are replenished.
    ///
    /// **The burst_size must not be zero.**
    pub burst_size: u32,
    /// Set the interval after which one user-initiated DHT operation quota item
    /// is replenished in seconds.
    ///
    /// When omitted from configuration, this defaults to [`Self::per_second`].
    ///
    /// **The interval must not be zero.**
    pub user_dht_per_second: u64,
    /// Set quota size for DHT operations initiated by HTTP user requests.
    ///
    /// When omitted from configuration, this defaults to [`Self::burst_size`].
    ///
    /// **The burst_size must not be zero.**
    pub user_dht_burst_size: u32,
}

#[derive(Deserialize)]
struct RateLimiterConfigToml {
    behind_proxy: bool,
    per_second: u64,
    burst_size: u32,
    #[serde(default)]
    user_dht_per_second: Option<u64>,
    #[serde(default)]
    user_dht_burst_size: Option<u32>,
}

impl<'de> Deserialize<'de> for RateLimiterConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let config = RateLimiterConfigToml::deserialize(deserializer)?;

        Ok(Self {
            behind_proxy: config.behind_proxy,
            per_second: config.per_second,
            burst_size: config.burst_size,
            user_dht_per_second: config.user_dht_per_second.unwrap_or(config.per_second),
            user_dht_burst_size: config.user_dht_burst_size.unwrap_or(config.burst_size),
        })
    }
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            behind_proxy: false,
            per_second: 2,
            burst_size: 10,
            user_dht_per_second: 2,
            user_dht_burst_size: 10,
        }
    }
}

#[derive(Debug, Clone)]
/// A rate limiter keyed by the request's normalized real IP.
pub struct IpRateLimiter {
    governor_middleware: Arc<GovernorConfig<RealIpKeyExtractor, StateInformationMiddleware>>,
}

impl IpRateLimiter {
    /// Create an [IpRateLimiter]
    ///
    /// This spawns a background task to clean up the rate limiting cache.
    pub async fn new(config: &RateLimiterConfig) -> Self {
        let governor_middleware = Arc::new(
            GovernorConfigBuilder::default()
                .use_headers()
                .per_second(config.per_second)
                .burst_size(config.burst_size)
                .key_extractor(RealIpKeyExtractor)
                .finish()
                .expect("failed to build rate-limiting governor"),
        );

        // The governor needs a background task for garbage collection (to clear expired records)
        let gc_interval = Duration::from_secs(60);

        let governor_limiter = governor_middleware.limiter().clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(gc_interval).await;
                tracing::debug!("rate limiting storage size: {}", governor_limiter.len());
                governor_limiter.retain_recent();
            }
        });

        Self {
            governor_middleware,
        }
    }

    /// Check if the Ip is allowed to make more requests
    pub fn is_limited(&self, ip: &IpAddr) -> bool {
        self.governor_middleware.limiter().check_key(ip).is_err()
    }

    /// Add a [GovernorLayer] on the provided [Router]
    pub fn layer(self, router: Router) -> Router {
        router.layer(GovernorLayer::new(self.governor_middleware))
    }
}

#[derive(Debug, Clone)]
/// A rate limiter for user-initiated DHT operations keyed by normalized real IP.
pub struct UserDhtRateLimiter {
    governor_middleware: Arc<GovernorConfig<RealIpKeyExtractor, NoOpMiddleware>>,
}

impl UserDhtRateLimiter {
    /// Create a [UserDhtRateLimiter].
    ///
    /// This spawns a background task to clean up the rate limiting cache.
    pub async fn new(config: &RateLimiterConfig) -> Self {
        let governor_middleware = Arc::new(
            GovernorConfigBuilder::default()
                .per_second(config.user_dht_per_second)
                .burst_size(config.user_dht_burst_size)
                .key_extractor(RealIpKeyExtractor)
                .finish()
                .expect("failed to build user DHT rate-limiting governor"),
        );

        let gc_interval = Duration::from_secs(60);

        let governor_limiter = governor_middleware.limiter().clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(gc_interval).await;
                tracing::debug!(
                    "user DHT rate limiting storage size: {}",
                    governor_limiter.len()
                );
                governor_limiter.retain_recent();
            }
        });

        Self {
            governor_middleware,
        }
    }

    /// Check if the IP is allowed to initiate more DHT operations.
    pub fn is_limited(&self, ip: &IpAddr) -> bool {
        self.governor_middleware.limiter().check_key(ip).is_err()
    }
}

impl pkarr::mainline::RequestFilter for IpRateLimiter {
    fn allow_request(
        &self,
        _request: &pkarr::mainline::RequestSpecific,
        from: std::net::SocketAddrV4,
    ) -> bool {
        !self.is_limited(&IpAddr::from(*from.ip()))
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::{RateLimiterConfig, UserDhtRateLimiter};

    #[tokio::test]
    async fn user_dht_limiter_uses_configured_quota() {
        let config = RateLimiterConfig {
            user_dht_per_second: 60,
            user_dht_burst_size: 1,
            ..Default::default()
        };
        let limiter = UserDhtRateLimiter::new(&config).await;
        let ip = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10));

        assert!(!limiter.is_limited(&ip));
        assert!(limiter.is_limited(&ip));
    }

    #[test]
    fn user_dht_limits_default_when_missing_from_toml() {
        let config: RateLimiterConfig = toml::from_str(
            r#"
behind_proxy = false
per_second = 7
burst_size = 13
"#,
        )
        .expect("legacy rate limiter config should deserialize");

        assert_eq!(config.user_dht_per_second, 7);
        assert_eq!(config.user_dht_burst_size, 13);
    }
}
