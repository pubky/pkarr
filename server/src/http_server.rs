use anyhow::Result;
use axum::extract::DefaultBodyLimit;
use axum::{http::Method, routing::get, Router};
use std::net::SocketAddr;
use tokio::{net::TcpListener, task::JoinSet};
use tower_http::cors::{self, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use pkarr::PkarrClientAsync;

use crate::rate_limiting::IpRateLimiter;

pub struct HttpServer {
    tasks: JoinSet<std::io::Result<()>>,
}

impl HttpServer {
    /// Spawn the server
    pub async fn spawn(
        client: PkarrClientAsync,
        port: u16,
        rate_limiter: IpRateLimiter,
    ) -> Result<HttpServer> {
        let app = create_app(AppState { client }, rate_limiter);

        let mut tasks = JoinSet::new();

        // launch http
        let app = app.clone();

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port)))
            .await?
            .into_std()?;
        let bound_addr = listener.local_addr()?;

        let fut = axum_server::from_tcp(listener)
            .serve(app.into_make_service_with_connect_info::<SocketAddr>());

        info!("HTTP server listening on {bound_addr}");

        tasks.spawn(fut);

        Ok(HttpServer { tasks })
    }

    /// Shutdown the server and wait for all tasks to complete.
    pub async fn shutdown(mut self) -> Result<()> {
        // TODO: Graceful cancellation.
        self.tasks.abort_all();
        self.run_until_done().await?;
        Ok(())
    }

    /// Wait for all tasks to complete.
    ///
    /// Runs forever unless tasks fail.
    pub async fn run_until_done(mut self) -> Result<()> {
        let mut final_res: anyhow::Result<()> = Ok(());
        while let Some(res) = self.tasks.join_next().await {
            match res {
                Ok(Ok(())) => {}
                Err(err) if err.is_cancelled() => {}
                Ok(Err(err)) => {
                    warn!(?err, "task failed");
                    final_res = Err(anyhow::Error::from(err));
                }
                Err(err) => {
                    warn!(?err, "task panicked");
                    final_res = Err(err.into());
                }
            }
        }
        final_res
    }
}

pub fn create_app(state: AppState, rate_limiter: IpRateLimiter) -> Router {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::PUT])
        .allow_origin(cors::Any);

    let router = Router::new()
        .route("/:key", get(crate::handlers::get).put(crate::handlers::put))
        .route(
            "/",
            get(|| async { "This is a Pkarr relay: pkarr.org/relays.\n" }),
        )
        .with_state(state)
        .layer(DefaultBodyLimit::max(1104))
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    rate_limiter.layer(router)
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub client: PkarrClientAsync,
}
