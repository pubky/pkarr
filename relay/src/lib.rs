//! A server that functions as a [pkarr](https://github.com/pubky/pkarr/) [relay](https://github.com/pubky/pkarr/blob/main/design/relays.md).
//!
//! You can run this relay as a binary or a crate for testing purposes.
//!

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod config;
mod error;
mod extractors;
mod handlers;
mod quota_config;
mod rate_limiting;
mod real_ip;
mod response;

use std::{
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::anyhow;
use axum::{extract::DefaultBodyLimit, Router};
use axum_server::Handle;

use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use pkarr::{extra::lmdb_cache::LmdbCache, mainline::async_dht::AsyncDht, Timestamp};
use url::Url;

use config::{Config, CACHE_DIR};

pub use quota_config::{RequestCountQuota, TimeUnit};
pub use rate_limiting::RateLimiterConfig;

/// A builder for Pkarr [Relay]
pub struct RelayBuilder(Config);

impl RelayBuilder {
    /// Set the port for the HTTP endpoint.
    pub fn http_port(&mut self, port: u16) -> &mut Self {
        self.0.http_port = port;

        self
    }

    /// Set the storage directory.
    ///
    /// This Relay's cache will be stored in a subdirectory (`pkarr-cache`) inside
    /// that storage directory
    ///
    /// Defaults to the path to the user's data directory
    pub fn storage(&mut self, storage: PathBuf) -> &mut Self {
        self.0.cache_path = Some(storage.join(CACHE_DIR));

        self
    }

    /// See [pkarr::ClientBuilder::cache_size]
    ///
    /// Defaults to `1_000_000`
    pub fn cache_size(&mut self, size: usize) -> &mut Self {
        self.0.cache_size = size;

        self
    }

    /// Disable the rate limiter.
    ///
    /// Useful when running in a local test network.
    pub fn disable_rate_limiter(&mut self) -> &mut Self {
        self.0.rate_limiter = None;

        self
    }

    /// Set the [RateLimiterConfig].
    ///
    /// Defaults to [RateLimiterConfig::default].
    pub fn rate_limiter_config(&mut self, config: RateLimiterConfig) -> &mut Self {
        self.0.rate_limiter = Some(config);

        self
    }

    /// Allows mutating the internal [pkarr::ClientBuilder] with a callback function.
    pub fn pkarr<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&mut pkarr::ClientBuilder) -> &mut pkarr::ClientBuilder,
    {
        f(&mut self.0.pkarr);

        self
    }

    /// Run a Pkarr relay with the provided configuration.
    ///
    /// # Safety
    /// This method is marked as unsafe because it uses `LmdbCache`, which can lead to
    /// undefined behavior if the lock file is corrupted or improperly handled.
    pub async unsafe fn run(self) -> anyhow::Result<Relay> {
        unsafe { Relay::run(self.0) }.await
    }
}

/// A running instance of a Pkarr relay server.
///
/// This struct represents a running relay server and provides methods to interact with it,
/// such as retrieving the server's address or shutting it down.
pub struct Relay {
    handle: Handle<SocketAddr>,
    relay_address: SocketAddr,
}

impl Relay {
    /// Run a Pkarr relay with the provided configuration.
    ///
    /// # Safety
    /// This method is marked as unsafe because it uses `LmdbCache`, which can lead to
    /// undefined behavior if the lock file is corrupted or improperly handled.
    ///
    /// # Arguments
    /// * `config` - The configuration for the relay.
    ///
    /// # Returns
    /// A `Result` containing the `Relay` instance or an error.
    async unsafe fn run(mut config: Config) -> anyhow::Result<Self> {
        tracing::debug!(?config, "Pkarr server config");

        let cache_path = config
            .cache_path
            .unwrap_or_else(|| {
                tracing::warn!("Cache path is not configured, running ephemeral Relay");
                std::env::temp_dir().join(Timestamp::now().to_string())
            })
            .join(CACHE_DIR);

        let cache = Arc::new(LmdbCache::open(&cache_path, config.cache_size)?);

        let behind_proxy = config
            .rate_limiter
            .as_ref()
            .map(|rate_limiter| rate_limiter.behind_proxy)
            .unwrap_or(false);

        let (rate_limiter, user_dht_rate_limiter) = match config.rate_limiter.as_ref() {
            Some(rate_limiter_config) => {
                let rate_limiter = rate_limiting::IpRateLimiter::new(rate_limiter_config)
                    .await
                    .map_err(|e| anyhow!("Failed to build IpRateLimiter: {e}"))?;
                let user_dht_rate_limiter =
                    rate_limiting::UserDhtRateLimiter::new(rate_limiter_config)
                        .await
                        .map_err(|e| anyhow!("Failed to build UserDhtRateLimiter: {e}"))?;
                config.pkarr.dht(|builder| {
                    builder.server_settings(pkarr::mainline::ServerSettings {
                        filter: Box::new(rate_limiter.clone()),
                        ..Default::default()
                    })
                });
                (Some(rate_limiter), Some(user_dht_rate_limiter))
            }
            None => (None, None),
        };

        let client = config.pkarr.build()?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.http_port)))?;
        // On axum-server 0.8.0 the `.set_nonblocking(true)` call does not take place internally anymore
        // See open issue https://github.com/programatik29/axum-server/issues/181
        listener.set_nonblocking(true)?;

        let dht = client.dht().expect("dht network is enabled").as_async();
        let node_address = dht.info().await.local_addr();
        let relay_address = listener.local_addr()?;

        info!("Cache path: {:?}", cache_path);
        info!("Running as a DHT node on {node_address}");
        info!("Running as a relay on TCP socket {relay_address}");

        let state = AppState {
            minimum_ttl: config.minimum_ttl,
            maximum_ttl: config.maximum_ttl,
            cache,
            cache_write_lock: Arc::new(Mutex::new(())),
            user_dht_rate_limiter,
            dht,
        };
        let app = create_app(state, rate_limiter, behind_proxy);

        let handle = Handle::new();

        let task = axum_server::from_tcp(listener)?
            .handle(handle.clone())
            .serve(app.into_make_service_with_connect_info::<SocketAddr>());

        tokio::spawn(task);

        Ok(Relay {
            handle,
            relay_address,
        })
    }

    /// Create a builder for running a [Relay]
    pub fn builder() -> RelayBuilder {
        RelayBuilder(Default::default())
    }

    /// Run a [Relay] with a configuration file path.
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn run_with_config_file(
        config_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        unsafe { Self::run(Config::load(config_path).await?) }.await
    }

    /// Run an ephemeral Pkarr relay on a random port number for testing purposes.
    /// Binds to `127.0.0.1`.
    /// # Arguments
    /// * `testnet` - A reference to a `mainline::Testnet` for bootstrapping the DHT.
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async fn run_test(testnet: &pkarr::mainline::Testnet) -> anyhow::Result<Self> {
        let mut config = Config {
            cache_path: None,
            http_port: 0,
            rate_limiter: None,
            ..Default::default()
        };

        config
            .pkarr
            .bootstrap(&testnet.bootstrap)
            .request_timeout(Duration::from_millis(100))
            .bootstrap(&testnet.bootstrap)
            .dht(|builder| {
                builder
                    .server_mode()
                    .bind_address(std::net::Ipv4Addr::LOCALHOST)
            });

        Ok(unsafe { Self::run(config).await? })
    }

    /// Run a Pkarr relay in a Testnet mode (on port 15411).
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn run_testnet() -> anyhow::Result<Self> {
        let testnet = pkarr::mainline::Testnet::builder(10).build()?;

        // Leaking the testnet to avoid dropping and shutting them down.
        for node in testnet.nodes {
            Box::leak(Box::new(node));
        }

        let mut config = Config {
            http_port: 15411,
            cache_path: None,
            rate_limiter: None,
            ..Default::default()
        };

        config
            .pkarr
            .request_timeout(Duration::from_millis(100))
            .bootstrap(&testnet.bootstrap)
            .dht(|builder| {
                builder
                    .server_mode()
                    .bind_address(std::net::Ipv4Addr::LOCALHOST)
            });

        Self::run(config).await
    }

    /// Returns the HTTP socket address of the relay.
    pub fn relay_address(&self) -> SocketAddr {
        self.relay_address
    }

    /// Returns the localhost URL of the HTTP server.
    pub fn local_url(&self) -> Url {
        Url::parse(&format!("http://localhost:{}", self.relay_address.port()))
            .expect("local_url should be formatted fine")
    }

    /// Shutdown the relay server.
    pub fn shutdown(&self) {
        self.handle.shutdown();
    }
}

fn create_app(
    state: AppState,
    rate_limiter: Option<rate_limiting::IpRateLimiter>,
    behind_proxy: bool,
) -> Router {
    let mut router = Router::new()
        .route(
            "/{key}",
            axum::routing::get(crate::handlers::get).put(crate::handlers::put),
        )
        .route("/", axum::routing::get(crate::handlers::index))
        .with_state(state)
        .layer(DefaultBodyLimit::max(1104))
        .layer(CorsLayer::very_permissive())
        .layer(TraceLayer::new_for_http());

    if let Some(rate_limiter) = rate_limiter {
        router = rate_limiter
            .layer(router)
            .layer(real_ip::middleware(behind_proxy));
    }

    router
}

#[derive(Debug, Clone)]
struct AppState {
    minimum_ttl: u32,
    maximum_ttl: u32,
    cache: Arc<LmdbCache>,
    // Synchronous mutex is fine here; handlers only hold it across cache reads/writes, never await points.
    cache_write_lock: Arc<Mutex<()>>,
    // Rate limiter for DHT operations initiated by HTTP user requests.
    user_dht_rate_limiter: Option<rate_limiting::UserDhtRateLimiter>,
    dht: AsyncDht,
}

#[cfg(test)]
mod tests {
    use std::{net::Ipv4Addr, num::NonZeroU32, time::Duration};

    use http::StatusCode;
    use pkarr::{Keypair, SignedPacket, Timestamp};

    use super::{RateLimiterConfig, Relay, RequestCountQuota};

    #[tokio::test]
    async fn user_dht_rate_limit_only_blocks_dht_operations() {
        let testnet = pkarr::mainline::Testnet::builder(2).build().unwrap();
        let relay = run_rate_limited_relay(&testnet).await;
        let http = reqwest::Client::new();
        let keypair = Keypair::random();
        let signed_packet = signed_packet(&keypair);
        let relay_url = relay.local_url();

        let put_response = http
            .put(relay_url.join(&keypair.public_key().to_string()).unwrap())
            .body(signed_packet.to_relay_payload())
            .send()
            .await
            .unwrap();
        assert_eq!(put_response.status(), StatusCode::NO_CONTENT);

        let local_cache_response = http
            .get(
                relay_url
                    .join(&format!(
                        "{}?policy=LocalOrRelayCacheOnly",
                        keypair.public_key()
                    ))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(local_cache_response.status(), StatusCode::OK);

        let cache_first_response = http
            .get(
                relay_url
                    .join(&format!("{}?policy=CacheFirst", keypair.public_key()))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(cache_first_response.status(), StatusCode::OK);

        let dht_response = http
            .get(
                relay_url
                    .join(&format!("{}?policy=DhtNetworkOnly", keypair.public_key()))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(dht_response.status(), StatusCode::TOO_MANY_REQUESTS);

        relay.shutdown();
    }

    async fn run_rate_limited_relay(testnet: &pkarr::mainline::Testnet) -> Relay {
        let storage = std::env::temp_dir().join(format!("pkarr-relay-{}", Timestamp::now()));
        let mut builder = Relay::builder();
        builder
            .cache_size(10)
            .http_port(0)
            .storage(storage)
            .rate_limiter_config(RateLimiterConfig {
                behind_proxy: false,
                quota: "60r/s".parse::<RequestCountQuota>().unwrap(),
                burst: NonZeroU32::new(100),
                user_dht_quota: "1r/h".parse::<RequestCountQuota>().unwrap(),
                user_dht_burst: NonZeroU32::new(1),
            })
            .pkarr(|builder| {
                builder
                    .no_default_network()
                    .bootstrap(&testnet.bootstrap)
                    .request_timeout(Duration::from_millis(100))
                    .dht(|builder| builder.bind_address(Ipv4Addr::LOCALHOST))
            });

        unsafe { builder.run().await.unwrap() }
    }

    fn signed_packet(keypair: &Keypair) -> SignedPacket {
        SignedPacket::builder()
            .txt(
                "example.com".try_into().unwrap(),
                "rate-limited".try_into().unwrap(),
                300,
            )
            .sign(keypair)
            .unwrap()
    }
}
