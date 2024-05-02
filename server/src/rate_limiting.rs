use std::{sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use governor::middleware::StateInformationMiddleware;
use tower_governor::{
    governor::GovernorConfigBuilder, key_extractor::PeerIpKeyExtractor, GovernorLayer,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct RateLimiterConfig {
    pub(crate) per_second: u64,
    pub(crate) burst_size: u32,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            per_second: 2,
            burst_size: 10,
        }
    }
}

pub type RateLimiterLayer = GovernorLayer<PeerIpKeyExtractor, StateInformationMiddleware>;

/// Create the default rate-limiting layer.
///
/// This will be used by the [crate::http_server::HttpServer] to guard all endpoints (GET and PUT)
/// and in [crate::dht_server::DhtServer] before calling [pkarr::client::mainline::rpc::Rpc::get]
/// after a cache miss or if its cached packet is expired.
///
/// The purpose is to limit DHT queries as much as possible, while serving honest clients still.
///
/// This spawns a background thread to clean up the rate limiting cache.
///
/// # Limits
///
/// * allow a burst of `10 requests` per IP address
/// * replenish `1 request` every `2 seconds`
pub fn create(config: &RateLimiterConfig) -> RateLimiterLayer {
    let governor_config = GovernorConfigBuilder::default()
        .use_headers()
        .per_second(config.per_second)
        .burst_size(config.burst_size)
        .finish()
        .expect("failed to build rate-limiting governor");

    let governor_config = Arc::new(governor_config);

    // The governor needs a background task for garbage collection (to clear expired records)
    let gc_interval = Duration::from_secs(60);
    let governor_limiter = governor_config.limiter().clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(gc_interval);
        tracing::debug!("rate limiting storage size: {}", governor_limiter.len());
        governor_limiter.retain_recent();
    });

    GovernorLayer {
        config: governor_config,
    }
}
