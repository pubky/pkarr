use std::{net::IpAddr, sync::Arc, time::Duration};

use axum::Router;
use governor::middleware::StateInformationMiddleware;
use serde::{Deserialize, Serialize};

use tower_governor::{
    governor::{GovernorConfig, GovernorConfigBuilder},
    key_extractor::{PeerIpKeyExtractor, SmartIpKeyExtractor},
};

pub use tower_governor::GovernorLayer;

#[derive(Serialize, Deserialize, Debug)]
pub struct RateLimiterConfig {
    pub(crate) behind_proxy: bool,
    pub(crate) per_second: u64,
    pub(crate) burst_size: u32,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            behind_proxy: false,
            per_second: 2,
            burst_size: 10,
        }
    }
}

#[derive(Debug, Clone)]
/// A rate limiter that works for direct connections (Peer) or behind reverse-proxy (Proxy)
pub enum IpRateLimiter {
    Peer(Arc<GovernorConfig<PeerIpKeyExtractor, StateInformationMiddleware>>),
    Proxy(Arc<GovernorConfig<SmartIpKeyExtractor, StateInformationMiddleware>>),
}

impl IpRateLimiter {
    /// Create an [IpRateLimiter]
    ///
    /// This spawns a background thread to clean up the rate limiting cache.
    pub fn new(config: &RateLimiterConfig) -> Self {
        match config.behind_proxy {
            true => {
                let config = Arc::new(
                    GovernorConfigBuilder::default()
                        .use_headers()
                        .per_second(config.per_second)
                        .burst_size(config.burst_size)
                        .key_extractor(SmartIpKeyExtractor)
                        .finish()
                        .expect("failed to build rate-limiting governor"),
                );

                // The governor needs a background task for garbage collection (to clear expired records)
                let gc_interval = Duration::from_secs(60);

                let governor_limiter = config.limiter().clone();
                std::thread::spawn(move || loop {
                    std::thread::sleep(gc_interval);
                    tracing::debug!("rate limiting storage size: {}", governor_limiter.len());
                    governor_limiter.retain_recent();
                });

                Self::Proxy(config)
            }
            false => {
                let config = Arc::new(
                    GovernorConfigBuilder::default()
                        .use_headers()
                        .per_second(config.per_second)
                        .burst_size(config.burst_size)
                        .finish()
                        .expect("failed to build rate-limiting governor"),
                );

                // The governor needs a background task for garbage collection (to clear expired records)
                let gc_interval = Duration::from_secs(60);

                let governor_limiter = config.limiter().clone();
                std::thread::spawn(move || loop {
                    std::thread::sleep(gc_interval);
                    tracing::debug!("rate limiting storage size: {}", governor_limiter.len());
                    governor_limiter.retain_recent();
                });

                Self::Peer(config)
            }
        }
    }

    /// Check if the Ip is allowed to make more requests
    pub fn is_limited(&self, ip: &IpAddr) -> bool {
        match self {
            IpRateLimiter::Peer(config) => config.limiter().check_key(ip).is_err(),
            IpRateLimiter::Proxy(config) => config.limiter().check_key(ip).is_err(),
        }
    }

    /// Add a [GovernorLayer] on the provided [Router]
    pub fn layer(&self, router: &Router) {
        let _ = match self {
            IpRateLimiter::Peer(config) => router.clone().layer(GovernorLayer {
                config: config.clone(),
            }),
            IpRateLimiter::Proxy(config) => router.clone().layer(GovernorLayer {
                config: config.clone(),
            }),
        };
    }
}
