//! A server that functions as a [pkarr](https://pkarr.org) [relay](https://pkarr.org/relays).
//!
//! You can run this relay as a binary or a crate for testing purposes.
//!

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod config;
mod error;
mod handlers;
mod rate_limiting;

use std::{
    net::{SocketAddr, TcpListener},
    sync::Arc,
    time::Duration,
};

use axum::{extract::DefaultBodyLimit, Router};
use axum_server::Handle;

use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use pkarr::{extra::lmdb_cache::LmdbCache, Client};
use url::Url;

pub use config::Config;

/// A running instance of a Pkarr relay server.
///
/// This struct represents a running relay server and provides methods to interact with it,
/// such as retrieving the server's address or shutting it down.
pub struct Relay {
    handle: Handle,
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
    pub async unsafe fn run(config: Config) -> anyhow::Result<Self> {
        let mut config = config;

        tracing::debug!(?config, "Pkarr server config");

        let cache_path = match config.cache_path {
            Some(path) => path,
            None => {
                let path = dirs_next::data_dir().ok_or_else(|| {
                    anyhow::anyhow!(
                        "operating environment provides no directory for application data"
                    )
                })?;
                path.join("pkarr-relay")
            }
        };

        let cache = Arc::new(LmdbCache::open(&cache_path, config.cache_size)?);

        let rate_limiter = config
            .rate_limiter
            .map(|rate_limiter| rate_limiting::IpRateLimiter::new(&rate_limiter));

        config.pkarr.cache(cache);

        if let Some(ref rate_limiter) = rate_limiter {
            config.pkarr.dht(|builder| {
                builder.server_settings(mainline::ServerSettings {
                    filter: Box::new(rate_limiter.clone()),
                    ..Default::default()
                })
            });
        }

        let client = config.pkarr.build()?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.http_port)))?;

        let node_address = client
            .dht()
            .expect("dht network is enabled")
            .info()
            .local_addr();
        let relay_address = listener.local_addr()?;

        info!("Running as a DHT node on {node_address}");
        info!("Running as a relay on TCP socket {relay_address}");

        let app = create_app(AppState { client }, rate_limiter);

        let handle = Handle::new();

        let task = axum_server::from_tcp(listener)
            .handle(handle.clone())
            .serve(app.into_make_service_with_connect_info::<SocketAddr>());

        tokio::spawn(task);

        Ok(Relay {
            handle,
            relay_address,
        })
    }

    /// Convenient wrapper around [`Self::run`].
    ///
    /// # Safety
    /// See [`Self::run`].
    pub async fn run_unsafe(config: Config) -> anyhow::Result<Self> {
        unsafe { Self::run(config).await }
    }

    /// Run an ephemeral Pkarr relay on a random port number for testing purposes.
    ///
    /// # Arguments
    /// * `testnet` - A reference to a `mainline::Testnet` for bootstrapping the DHT.
    ///
    /// # Safety
    /// See [`Self::run`].
    pub async fn run_test(testnet: &mainline::Testnet) -> anyhow::Result<Self> {
        let storage = std::env::temp_dir().join(pubky_timestamp::Timestamp::now().to_string());

        let mut config = Config {
            cache_path: Some(storage.join("pkarr-relay")),
            http_port: 0,
            ..Default::default()
        };

        config
            .pkarr
            .bootstrap(&testnet.bootstrap)
            .request_timeout(Duration::from_millis(100))
            .bootstrap(&testnet.bootstrap)
            .dht(|builder| builder.server_mode());

        Ok(unsafe { Self::run(config).await? })
    }

    /// Run a Pkarr relay in a Testnet mode (on port 15411).
    ///
    /// # Safety
    /// See [`Self::run`].
    pub async fn run_testnet() -> anyhow::Result<Self> {
        let testnet = mainline::Testnet::new(10)?;

        // Leaking the testnet to avoid dropping and shutting them down.
        for node in testnet.nodes {
            Box::leak(Box::new(node));
        }

        let storage = std::env::temp_dir().join(pubky_timestamp::Timestamp::now().to_string());

        let mut config = Config {
            http_port: 15411,
            cache_path: Some(storage.join("pkarr-relay")),
            rate_limiter: None,
            ..Default::default()
        };

        config
            .pkarr
            .request_timeout(Duration::from_millis(100))
            .bootstrap(&testnet.bootstrap)
            .dht(|builder| builder.server_mode());

        unsafe { Self::run(config).await }
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

fn create_app(state: AppState, rate_limiter: Option<rate_limiting::IpRateLimiter>) -> Router {
    let mut router = Router::new()
        .route(
            "/{key}",
            axum::routing::get(crate::handlers::get).put(crate::handlers::put),
        )
        .route(
            "/",
            axum::routing::get(|| async { "This is a Pkarr relay: pkarr.org/relays.\n" }),
        )
        .with_state(state)
        .layer(DefaultBodyLimit::max(1104))
        .layer(CorsLayer::very_permissive())
        .layer(TraceLayer::new_for_http());

    if let Some(rate_limiter) = rate_limiter {
        router = rate_limiter.layer(router);
    }

    router
}

#[derive(Debug, Clone)]
struct AppState {
    /// The Pkarr client for DHT operations.
    client: Client,
}
