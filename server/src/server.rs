mod handlers;

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{ConnectInfo, Request},
    handler::Handler,
    http::Method,
    middleware::{self, Next},
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::{net::TcpListener, task::JoinSet};
use tower_http::{
    cors::{self, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, span, warn, Level};

use pkarr::{async_client::AsyncPkarrClient, PkarrClient};

/// The HTTP(S) server part of iroh-dns-server
pub struct HttpServer {
    tasks: JoinSet<std::io::Result<()>>,
}

impl HttpServer {
    /// Spawn the server
    pub async fn spawn(client: AsyncPkarrClient) -> Result<HttpServer> {
        let app = create_app(AppState { client });

        let mut tasks = JoinSet::new();

        // launch http
        let app = app.clone();

        let listener = TcpListener::bind("0.0.0.0:6881").await?.into_std()?;
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

pub(crate) fn create_app(state: AppState) -> Router {
    // configure cors middleware
    let cors = CorsLayer::new()
        // allow `GET` and `POST` when accessing the resource
        .allow_methods([Method::GET, Method::POST, Method::PUT])
        // allow requests from any origin
        .allow_origin(cors::Any);

    // configure tracing middleware
    // let trace = TraceLayer::new_for_http().make_span_with(|request: &http::Request<_>| {
    //     let conn_info = request
    //         .extensions()
    //         .get::<ConnectInfo<SocketAddr>>()
    //         .expect("connectinfo extension to be present");
    //     let span = span!(
    //     Level::DEBUG,
    //         "http_request",
    //         method = ?request.method(),
    //         uri = ?request.uri(),
    //         src = %conn_info.0,
    //         );
    //     span
    // });

    // configure rate limiting middleware
    // let rate_limit = rate_limiting::create();

    // configure routes
    //
    // only the pkarr::put route gets a rate limit
    let router = Router::new()
        // .route("/dns-query", get(doh::get).post(doh::post))
        .route(
            "/:key",
            get(handlers::get).put(
                handlers::put, // Add rate limiting
                               // .layer(rate_limit)
            ),
        )
        .route("/ping", get(|| async { "Pong" }))
        .route(
            "/",
            get(|| async { "This is a Pkarr relay: pkarr.org/relays.\n" }),
        )
        .with_state(state);

    // configure app
    router.layer(cors)
    // .layer(trace)
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub client: AsyncPkarrClient,
}
