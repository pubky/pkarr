mod config;
mod dht_server;
mod error;
mod handlers;
mod rate_limiting;

use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
};

use axum::{extract::DefaultBodyLimit, Router};
use axum_server::Handle;

use dht_server::DhtServer;
use rate_limiting::RateLimiterConfig;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use pkarr::{extra::lmdb_cache::LmdbCache, Client};

pub use config::Config;

#[derive(Debug, Default)]
pub struct RelayBuilder(Config);

impl RelayBuilder {
    // Configure the port for the HTTP server to listen on
    pub fn http_port(mut self, port: u16) -> Self {
        self.0.http_port = port;

        self
    }

    // Configure the port for the internal Mainline DHT node to listen on
    pub fn dht_port(mut self, port: u16) -> Self {
        self.0.pkarr_config.dht_config.port = Some(port);

        self
    }

    // Configure the path to store the persistent cache at.
    pub fn cache_path(mut self, path: PathBuf) -> Self {
        self.0.cache_path = Some(path);

        self
    }

    // Configure the maximum number of SignedPackets in the LRU cache
    pub fn cache_size(mut self, size: usize) -> Self {
        self.0.cache_size = size;

        self
    }

    /// Disable rate limiting by setting the configuration as generous as possible
    pub fn disable_rate_limiting(mut self) -> Self {
        self.0.rate_limiter = RateLimiterConfig {
            per_second: 1,
            burst_size: u32::MAX,
            behind_proxy: false,
        };

        self
    }

    /// Start a Pkarr [relay](https://pkarr.org/relays) http server as well as dht [resolver](https://pkarr.org/resolvers).
    ///
    /// # Safety
    /// Relay uses LmdbCache, [opening][pkarr::extra::lmdb_cache::LmdbCache::open] which is marked unsafe,
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn start(self) -> anyhow::Result<Relay> {
        Ok(unsafe { Relay::start(self.0).await? })
    }
}

pub struct Relay {
    handle: Handle,
    resolver_address: SocketAddr,
    relay_address: SocketAddr,
}

impl Relay {
    pub fn builder() -> RelayBuilder {
        RelayBuilder::default()
    }

    /// Start a Pkarr [relay](https://pkarr.org/relays) http server as well as dht [resolver](https://pkarr.org/resolvers).
    ///
    /// # Safety
    /// Relay uses LmdbCache, [opening][pkarr::extra::lmdb_cache::LmdbCache::open] which is marked unsafe,
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn start(config: Config) -> anyhow::Result<Self> {
        let mut config = config;

        let cache_path = match config.cache_path {
            Some(path) => path,
            None => {
                let path = dirs_next::data_dir().ok_or_else(|| {
                    anyhow::anyhow!(
                        "operating environment provides no directory for application data"
                    )
                })?;
                path.join("pkarr-server")
            }
        };

        let cache = Box::new(LmdbCache::open(&cache_path, config.cache_size)?);

        let rate_limiter = rate_limiting::IpRateLimiter::new(&config.rate_limiter);

        let server = Box::new(DhtServer::new(
            cache.clone(),
            config.pkarr_config.resolvers.clone(),
            config.pkarr_config.minimum_ttl,
            config.pkarr_config.maximum_ttl,
            rate_limiter.clone(),
        ));

        config.pkarr_config.dht_config.server = Some(server);
        config.pkarr_config.cache = Some(cache);

        let client = Client::new(config.pkarr_config)?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.http_port)))?;

        let resolver_address = *client.info()?.local_addr()?;
        let relay_address = listener.local_addr()?;

        info!("Running as a resolver on UDP socket {resolver_address}");
        info!("Running as a relay on TCP socket {relay_address}");

        let app = create_app(AppState { client }, rate_limiter);

        let handle = Handle::new();

        let task = axum_server::from_tcp(listener)
            .handle(handle.clone())
            .serve(app.into_make_service_with_connect_info::<SocketAddr>());

        tokio::spawn(task);

        Ok(Relay {
            handle,
            resolver_address,
            relay_address,
        })
    }

    /// Convenient wrapper around [Self::start].
    ///
    /// Make sure to read the safety section in [Self::start]
    pub async fn start_unsafe(config: Config) -> anyhow::Result<Self> {
        unsafe { Self::start(config).await }
    }

    pub fn resolver_address(&self) -> SocketAddr {
        self.resolver_address
    }

    pub fn relay_address(&self) -> SocketAddr {
        self.relay_address
    }

    pub fn shutdown(&self) {
        self.handle.shutdown();
    }
}

pub fn create_app(state: AppState, rate_limiter: rate_limiting::IpRateLimiter) -> Router {
    let router = Router::new()
        .route(
            "/:key",
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

    rate_limiter.layer(router)
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub client: Client,
}
