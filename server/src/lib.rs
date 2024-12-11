mod config;
mod dht_server;
mod error;
mod handlers;
mod rate_limiting;

use std::net::{SocketAddr, TcpListener};

use axum::{extract::DefaultBodyLimit, Router};
use axum_server::Handle;

use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use pkarr::{extra::lmdb_cache::LmdbCache, mainline, Client};

pub use config::Config;

pub struct Relay {
    handle: Handle,
    resolver_address: SocketAddr,
    relay_address: SocketAddr,
}

impl Relay {
    /// Start a Pkarr [relay](https://pkarr.org/relays) http server as well as dht [resolver](https://pkarr.org/resolvers).
    ///
    /// # Safety
    /// Relay uses LmdbCache, [opening][pkarr::extra::lmdb_cache::LmdbCache::open] which is marked unsafe,
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn start(config: Config) -> anyhow::Result<Self> {
        let cache = Box::new(LmdbCache::open(&config.cache_path()?, config.cache_size())?);

        let rate_limiter = rate_limiting::IpRateLimiter::new(config.rate_limiter());

        let client = Client::builder()
            .dht_settings(
                mainline::Settings::default()
                    .port(config.dht_port())
                    .custom_server(Box::new(dht_server::DhtServer::new(
                        cache.clone(),
                        config.resolvers(),
                        config.minimum_ttl(),
                        config.maximum_ttl(),
                        rate_limiter.clone(),
                    ))),
            )
            .resolvers(config.resolvers())
            .minimum_ttl(config.minimum_ttl())
            .maximum_ttl(config.maximum_ttl())
            .cache(cache)
            .build()?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.relay_port())))?;

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
