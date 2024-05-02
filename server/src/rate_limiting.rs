use std::{sync::Arc, time::Duration};

use governor::{clock::QuantaInstant, middleware::NoOpMiddleware};
use tower_governor::{
    governor::GovernorConfigBuilder, key_extractor::PeerIpKeyExtractor, GovernorLayer,
};

/// Create the default rate-limiting layer.
///
/// This spawns a background thread to clean up the rate limiting cache.
pub fn create() -> GovernorLayer<PeerIpKeyExtractor, NoOpMiddleware<QuantaInstant>> {
    // Configure rate limiting:
    // * allow only one requests per IP address
    // * replenish one element every two seconds
    let governor_config = GovernorConfigBuilder::default()
        .per_second(2)
        .burst_size(1)
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
