use crate::config::{IpWhitelist, RelayMode};
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    Router,
};
use governor::middleware::StateInformationMiddleware;
use serde::{Deserialize, Serialize};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};
pub use tower_governor::GovernorLayer;
use tower_governor::{
    governor::{GovernorConfig, GovernorConfigBuilder},
    key_extractor::{PeerIpKeyExtractor, SmartIpKeyExtractor},
};

#[derive(Serialize, Deserialize, Debug)]
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
    pub fn layer(self, router: Router, mode: RelayMode, whitelist: IpWhitelist) -> Router {
        // If private mode, use custom middleware that respects whitelist
        if matches!(mode, RelayMode::PRIVATE) {
            return self.handle_private_mode(router, whitelist);
        }

        // Default behavior for legacy and public modes
        match self {
            IpRateLimiter::Peer(config) => router.layer(GovernorLayer {
                config: config.clone(),
            }),
            IpRateLimiter::Proxy(config) => router.layer(GovernorLayer {
                config: config.clone(),
            }),
        }
    }

    fn handle_private_mode(self, router: Router, whitelist: IpWhitelist) -> Router {
        let whitelist = whitelist;

        router.layer(middleware::from_fn(
            move |ConnectInfo(addr): ConnectInfo<SocketAddr>,
                  request: Request<Body>,
                  next: Next| {
                let rate_limiter = self.clone();
                let whitelist = whitelist.clone();
                async move {
                    // Check if IP is whitelisted (skip rate limiting if it is)
                    let is_whitelisted = whitelist.is_trusted(&addr.ip());

                    if !is_whitelisted && rate_limiter.is_limited(&addr.ip()) {
                        return (StatusCode::TOO_MANY_REQUESTS, "Too Many Requests")
                            .into_response();
                    }
                    next.run(request).await
                }
            },
        ))
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
