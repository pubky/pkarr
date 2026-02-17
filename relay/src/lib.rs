//! A server that functions as a [pkarr](https://github.com/pubky/pkarr/) [relay](https://github.com/pubky/pkarr/blob/main/design/relays.md).
//!
//! You can run this relay as a binary or a crate for testing purposes.
//!

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod config;
mod error;
mod handlers;
/// Operation-based rate limiting
pub mod rate_limiter;
/// IP-based rate limiting
pub mod rate_limiting;

use axum::{extract::DefaultBodyLimit, Router};
use axum_server::Handle;
pub use config::Config;
use config::CACHE_DIR;
use pkarr::{extra::lmdb_cache::LmdbCache, Client, Timestamp};
pub use rate_limiter::*;
use std::{
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;
use url::Url;

/// A builder for Pkarr [Relay]
pub struct RelayBuilder(Config);

impl RelayBuilder {
    /// Create a new RelayBuilder from a Config.
    pub fn new(config: Config) -> Self {
        RelayBuilder(config)
    }

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
        self.0.dht_rate_limiter = None;
        self.0.http_rate_limiter = None;

        self
    }

    /// Set the operation-based rate limiter configuration.
    ///
    /// Defaults to None (no operation-based rate limiting).
    pub fn rate_limiter_config(&mut self, limits: Option<Vec<OperationLimit>>) -> &mut Self {
        self.0.http_rate_limiter = limits;

        self
    }

    /// Set whether to respect the most_recent query parameter.
    ///
    /// Defaults to false.
    pub fn set_resolve_most_recent(&mut self, value: bool) -> &mut Self {
        self.0.resolve_most_recent = value;

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
    async unsafe fn run(config: Config) -> anyhow::Result<Self> {
        let mut config = config;

        tracing::debug!(?config, "Pkarr server config");

        let cache_path = config
            .cache_path
            .unwrap_or({
                tracing::warn!("Cache path is not configured, running ephemeral Relay");

                std::env::temp_dir().join(Timestamp::now().to_string())
            })
            .join(CACHE_DIR);

        let cache = Arc::new(LmdbCache::open(&cache_path, config.cache_size)?);

        let dht_rate_limiter = config.dht_rate_limiter.map(|mut dht_config| {
            // Override behind_proxy with global setting
            dht_config.behind_proxy = config.behind_proxy;
            rate_limiting::IpRateLimiter::new(&dht_config)
        });

        let http_rate_limiter = config.http_rate_limiter.clone();

        config.pkarr.cache(cache);

        // Apply DHT rate limiting (if enabled)
        if let Some(ref rate_limiter) = dht_rate_limiter {
            config.pkarr.dht(|builder| {
                builder.server_settings(pkarr::mainline::ServerSettings {
                    filter: Box::new(rate_limiter.clone()),
                    ..Default::default()
                })
            });
        }

        let client = config.pkarr.build()?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.http_port)))?;
        // On axum-server 0.8.0 the `.set_nonblocking(true)` call does not take place internally anymore
        // See open issue https://github.com/programatik29/axum-server/issues/181
        listener.set_nonblocking(true)?;

        let node_address = client
            .dht()
            .expect("dht network is enabled")
            .info()
            .local_addr();
        let relay_address = listener.local_addr()?;

        info!("Cache path: {:?}", cache_path);
        info!("Running as a DHT node on {node_address}");
        info!("Running as a relay on TCP socket {relay_address}");

        let app = create_app(AppState {
            client,
            resolve_most_recent: config.resolve_most_recent,
            dht_rate_limiter,
            http_rate_limiter,
            behind_proxy: config.behind_proxy,
        });

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
            dht_rate_limiter: None,
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
            dht_rate_limiter: None,
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

fn create_app(state: AppState) -> Router {
    // Extract values before state is moved
    let dht_rate_limiter = state.dht_rate_limiter.clone();
    let http_rate_limiter = state.http_rate_limiter.clone();
    let behind_proxy = state.behind_proxy;
    let resolve_most_recent = state.resolve_most_recent;

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

    // Apply operation-based rate limiting (if configured)
    if let Some(limits) = http_rate_limiter {
        router = router.layer(RateLimiterLayer::new(
            limits,
            behind_proxy,
            resolve_most_recent,
        ));
    } else if let Some(dht_limiter) = dht_rate_limiter {
        // Fallback to IP-based rate limiting for HTTP if no operation-based limits
        router = dht_limiter.layer(router);
    }

    router
}

#[derive(Debug, Clone)]
struct AppState {
    /// The Pkarr client for DHT operations.
    client: Client,
    /// Whether to respect the most_recent query parameter
    resolve_most_recent: bool,
    /// DHT rate limiter (IP-based, protects DHT node)
    dht_rate_limiter: Option<rate_limiting::IpRateLimiter>,
    /// HTTP rate limiter (operation-based, protects HTTP server)
    http_rate_limiter: Option<Vec<OperationLimit>>,
    /// Whether this relay is behind a proxy
    behind_proxy: bool,
}
