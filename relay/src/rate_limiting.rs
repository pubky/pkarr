use std::{net::IpAddr, num::NonZeroU32, sync::Arc, time::Duration};

use crate::quota_config::{RequestCountQuota, TimeUnit};
use crate::real_ip::RealIpKeyExtractor;
use axum::Router;
use governor::{
    clock::QuantaInstant,
    middleware::{NoOpMiddleware, RateLimitingMiddleware, StateInformationMiddleware},
};
use serde::{Deserialize, Serialize};
use tower_governor::governor::{GovernorConfig, GovernorConfigBuilder};
use tower_governor::key_extractor::KeyExtractor;
pub use tower_governor::GovernorLayer;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
/// Configurations for rate limitng.
pub struct RateLimiterConfig {
    /// Enable rate limit based on headers commonly used by reverse proxies.
    ///
    /// Uses headers commonly used by reverse proxies to extract the original IP address,
    /// falling back to the connection's peer IP address.
    /// <https://docs.rs/tower_governor/latest/tower_governor/key_extractor/struct.SmartIpKeyExtractor.html>
    pub behind_proxy: bool,

    /// HTTP request quota applied per normalized real IP.
    pub quota: RequestCountQuota,
    /// Optional HTTP request burst size.
    ///
    /// When omitted, this defaults to the rate configured in [`Self::quota`].
    #[serde(default)]
    pub burst: Option<NonZeroU32>,

    /// User-initiated DHT operation quota applied per normalized real IP.
    pub user_dht_quota: RequestCountQuota,
    /// Optional user-initiated DHT operation burst size.
    ///
    /// When omitted, this defaults to the rate configured in [`Self::user_dht_quota`].
    #[serde(default)]
    pub user_dht_burst: Option<NonZeroU32>,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            behind_proxy: false,
            quota: RequestCountQuota {
                rate: NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN),
                time_unit: TimeUnit::Second,
            },
            burst: NonZeroU32::new(10),
            user_dht_quota: RequestCountQuota {
                rate: NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN),
                time_unit: TimeUnit::Second,
            },
            user_dht_burst: NonZeroU32::new(10),
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
    pub async fn new(config: &RateLimiterConfig) -> Result<Self, String> {
        let quota = config.quota.to_governor_quota(config.burst)?;
        let builder = GovernorConfigBuilder::default()
            .use_headers()
            .key_extractor(RealIpKeyExtractor);
        let governor_middleware = build_with_quota(builder, &quota);
        let governor_middleware = Arc::new(governor_middleware);

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

        Ok(Self {
            governor_middleware,
        })
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
    pub async fn new(config: &RateLimiterConfig) -> Result<Self, String> {
        let quota = config
            .user_dht_quota
            .to_governor_quota(config.user_dht_burst)?;
        let builder = GovernorConfigBuilder::default().key_extractor(RealIpKeyExtractor);
        let governor_middleware = build_with_quota(builder, &quota);
        let governor_middleware = Arc::new(governor_middleware);

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

        Ok(Self {
            governor_middleware,
        })
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

fn build_with_quota<K, M>(
    mut builder: GovernorConfigBuilder<K, M>,
    quota: &governor::Quota,
) -> GovernorConfig<K, M>
where
    K: KeyExtractor,
    M: RateLimitingMiddleware<QuantaInstant>,
{
    builder
        .period(quota.replenish_interval())
        .burst_size(quota.burst_size().get())
        .finish()
        .expect("failed to build GovernorConfig from governor::Quota")
}

#[cfg(test)]
mod tests {
    use std::{
        net::{IpAddr, Ipv4Addr},
        num::NonZeroU32,
    };

    use super::{RateLimiterConfig, UserDhtRateLimiter};
    use crate::quota_config::RequestCountQuota;

    #[tokio::test]
    async fn user_dht_limiter_uses_configured_quota() {
        let config = RateLimiterConfig {
            user_dht_quota: "60r/s".parse::<RequestCountQuota>().unwrap(),
            user_dht_burst: NonZeroU32::new(1),
            ..Default::default()
        };
        let limiter = UserDhtRateLimiter::new(&config).await.unwrap();
        let ip = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10));

        assert!(!limiter.is_limited(&ip));
        assert!(limiter.is_limited(&ip));
    }

    #[test]
    fn rate_limiter_config_deserializes_new_quota_fields() {
        let config: RateLimiterConfig = toml::from_str(
            r#"
behind_proxy = false
quota = "7r/s"
burst = 13
user_dht_quota = "3r/m"
user_dht_burst = 5
"#,
        )
        .expect("rate limiter config should deserialize");

        assert_eq!(config.quota.to_string(), "7r/s");
        assert_eq!(config.burst, NonZeroU32::new(13));
        assert_eq!(config.user_dht_quota.to_string(), "3r/m");
        assert_eq!(config.user_dht_burst, NonZeroU32::new(5));
    }

    #[test]
    fn omitted_bursts_default_to_quota_rate() {
        let config: RateLimiterConfig = toml::from_str(
            r#"
behind_proxy = false
quota = "7r/s"
user_dht_quota = "3r/m"
"#,
        )
        .expect("rate limiter config should deserialize");

        assert_eq!(
            config
                .quota
                .to_governor_quota(config.burst)
                .unwrap()
                .burst_size(),
            NonZeroU32::new(7).unwrap()
        );
        assert_eq!(
            config
                .user_dht_quota
                .to_governor_quota(config.user_dht_burst)
                .unwrap()
                .burst_size(),
            NonZeroU32::new(3).unwrap()
        );
    }

    #[test]
    fn zero_burst_is_rejected() {
        let err = toml::from_str::<RateLimiterConfig>(
            r#"
behind_proxy = false
quota = "7r/s"
burst = 0
user_dht_quota = "3r/m"
"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains("invalid value"));
    }

    #[test]
    fn legacy_rate_limiter_fields_are_rejected() {
        let err = toml::from_str::<RateLimiterConfig>(
            r#"
behind_proxy = false
per_second = 7
burst_size = 13
"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains("unknown field"));
    }
}
