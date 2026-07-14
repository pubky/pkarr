use std::{net::IpAddr, num::NonZeroU32, sync::Arc, time::Duration};

use crate::quota_config::{RequestCountQuota, TimeUnit};
use crate::real_ip::RealIpKeyExtractor;
use axum::Router;
use governor::{
    clock::{QuantaClock, QuantaInstant},
    middleware::{RateLimitingMiddleware, StateInformationMiddleware},
    state::keyed::DashMapStateStore,
    RateLimiter,
};
use serde::{Deserialize, Serialize};
use tower_governor::governor::{GovernorConfig, GovernorConfigBuilder};
use tower_governor::GovernorLayer;

type IpAddrRateLimiter = RateLimiter<IpAddr, DashMapStateStore<IpAddr>, QuantaClock>;

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
    ///
    /// This is always applied before request handling to every HTTP request,
    /// including GET, PUT, the index route, cache hits, and cache-only requests.
    pub quota: RequestCountQuota,
    /// Optional HTTP request burst size.
    ///
    /// When omitted, this defaults to the rate configured in [`Self::quota`].
    #[serde(default)]
    pub burst: Option<NonZeroU32>,

    /// Incoming DHT request quota applied per DHT peer IP.
    ///
    /// This is enforced by the internal Mainline DHT node before handling
    /// requests from other DHT peers.
    pub dht_quota: RequestCountQuota,
    /// Optional incoming DHT request burst size.
    ///
    /// When omitted, this defaults to the rate configured in [`Self::dht_quota`].
    #[serde(default)]
    pub dht_burst: Option<NonZeroU32>,

    /// User-initiated DHT operation quota applied per normalized real IP.
    ///
    /// This is separate from [`Self::quota`] and is consumed only when a request
    /// makes the relay contact the DHT: PUT publishes, GET cache misses, expired
    /// `CacheFirst` entries, and `NetworkOnly` requests. Cache hits and
    /// `CacheOnly` requests do not consume this quota.
    pub user_dht_quota: RequestCountQuota,
    /// Optional user-initiated DHT operation burst size.
    ///
    /// When omitted, this defaults to the rate configured in [`Self::user_dht_quota`].
    #[serde(default)]
    pub user_dht_burst: Option<NonZeroU32>,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        let default_quota = RequestCountQuota {
            rate: NonZeroU32::new(2).expect("failed to build NonZeroU32"),
            time_unit: TimeUnit::Second,
        };
        Self {
            behind_proxy: false,
            quota: default_quota.clone(),
            burst: None,
            dht_quota: default_quota.clone(),
            dht_burst: None,
            user_dht_quota: default_quota,
            user_dht_burst: None,
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
        let governor_middleware = GovernorConfigBuilder::default()
            .use_headers()
            .key_extractor(RealIpKeyExtractor)
            .period(quota.replenish_interval())
            .burst_size(quota.burst_size().get())
            .finish()
            .expect("failed to build GovernorConfig from governor::Quota");
        let governor_middleware = Arc::new(governor_middleware);

        let governor_limiter = governor_middleware.limiter().clone();
        spawn_rate_limiter_gc("HTTP", governor_limiter);

        Ok(Self {
            governor_middleware,
        })
    }

    /// Add a [GovernorLayer] on the provided [Router]
    pub fn layer(self, router: Router) -> Router {
        router.layer(GovernorLayer::new(self.governor_middleware))
    }
}

#[derive(Debug, Clone)]
struct IpAddrLimiter {
    governor_limiter: Arc<IpAddrRateLimiter>,
}

impl IpAddrLimiter {
    fn new(label: &'static str, quota: governor::Quota) -> Self {
        let governor_limiter = Arc::new(RateLimiter::dashmap(quota));

        spawn_rate_limiter_gc(label, governor_limiter.clone());

        Self { governor_limiter }
    }

    fn is_allowed(&self, ip: &IpAddr) -> bool {
        self.governor_limiter.check_key(ip).is_ok()
    }
}

#[derive(Debug, Clone)]
/// A rate limiter for incoming DHT requests keyed by peer IP.
pub struct DhtRateLimiter {
    limiter: IpAddrLimiter,
}

impl DhtRateLimiter {
    /// Create a [DhtRateLimiter].
    ///
    /// This spawns a background task to clean up the rate limiting cache.
    pub async fn new(config: &RateLimiterConfig) -> Result<Self, String> {
        let quota = config.dht_quota.to_governor_quota(config.dht_burst)?;

        Ok(Self {
            limiter: IpAddrLimiter::new("DHT", quota),
        })
    }
}

impl pkarr::mainline::RequestFilter for DhtRateLimiter {
    fn allow_request(
        &self,
        _request: &pkarr::mainline::RequestSpecific,
        from: std::net::SocketAddrV4,
    ) -> bool {
        self.limiter.is_allowed(&IpAddr::from(*from.ip()))
    }
}

#[derive(Debug, Clone)]
/// A rate limiter for user-initiated DHT operations keyed by normalized real IP.
pub struct UserDhtRateLimiter {
    limiter: IpAddrLimiter,
}

impl UserDhtRateLimiter {
    /// Create a [UserDhtRateLimiter].
    ///
    /// This spawns a background task to clean up the rate limiting cache.
    pub async fn new(config: &RateLimiterConfig) -> Result<Self, String> {
        let quota = config
            .user_dht_quota
            .to_governor_quota(config.user_dht_burst)?;

        Ok(Self {
            limiter: IpAddrLimiter::new("user DHT", quota),
        })
    }

    /// Check if the IP is allowed to initiate more DHT operations.
    pub fn is_limited(&self, ip: &IpAddr) -> bool {
        !self.limiter.is_allowed(ip)
    }
}

fn spawn_rate_limiter_gc<K, M>(
    label: &'static str,
    governor_limiter: Arc<RateLimiter<K, DashMapStateStore<K>, QuantaClock, M>>,
) where
    K: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
    M: RateLimitingMiddleware<QuantaInstant> + Send + Sync + 'static,
{
    let gc_interval = Duration::from_secs(60);

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(gc_interval).await;
            tracing::debug!(
                "{label} rate limiting storage size: {}",
                governor_limiter.len()
            );
            governor_limiter.retain_recent();
        }
    });
}

#[cfg(test)]
mod tests {
    use std::{
        net::{IpAddr, Ipv4Addr},
        num::NonZeroU32,
    };

    use super::{DhtRateLimiter, RateLimiterConfig, UserDhtRateLimiter};
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

    #[tokio::test]
    async fn dht_limiter_implements_request_filter() {
        fn assert_request_filter<T: pkarr::mainline::RequestFilter>() {}
        assert_request_filter::<DhtRateLimiter>();

        let config = RateLimiterConfig {
            dht_quota: "60r/s".parse::<RequestCountQuota>().unwrap(),
            dht_burst: NonZeroU32::new(1),
            ..Default::default()
        };

        DhtRateLimiter::new(&config).await.unwrap();
    }

    #[test]
    fn rate_limiter_config_deserializes_new_quota_fields() {
        let config: RateLimiterConfig = toml::from_str(
            r#"
behind_proxy = false
quota = "7r/s"
burst = 13
dht_quota = "11r/s"
dht_burst = 17
user_dht_quota = "3r/m"
user_dht_burst = 5
"#,
        )
        .expect("rate limiter config should deserialize");

        assert_eq!(config.quota.to_string(), "7r/s");
        assert_eq!(config.burst, NonZeroU32::new(13));
        assert_eq!(config.dht_quota.to_string(), "11r/s");
        assert_eq!(config.dht_burst, NonZeroU32::new(17));
        assert_eq!(config.user_dht_quota.to_string(), "3r/m");
        assert_eq!(config.user_dht_burst, NonZeroU32::new(5));
    }

    #[test]
    fn omitted_bursts_default_to_quota_rate() {
        let config: RateLimiterConfig = toml::from_str(
            r#"
behind_proxy = false
quota = "7r/s"
dht_quota = "11r/s"
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
                .dht_quota
                .to_governor_quota(config.dht_burst)
                .unwrap()
                .burst_size(),
            NonZeroU32::new(11).unwrap()
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
dht_quota = "11r/s"
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
