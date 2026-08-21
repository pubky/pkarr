//! HTTP/1 and HTTP/2 serving with explicit connection resource bounds.
//!
//! A connection must produce its first complete request headers promptly and
//! has a maximum total age. At that age Hyper gracefully refuses new requests
//! (with GOAWAY for HTTP/2), lets active requests drain, and is forcibly
//! dropped if the drain timeout expires.

use std::{
    io,
    net::{SocketAddr, TcpListener},
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::{ConnectInfo, Request},
    middleware::map_request,
    Extension, Router,
};
use hyper_util::{
    rt::{TokioExecutor, TokioIo, TokioTimer},
    server::conn::auto,
    service::TowerToHyperService,
};
use tokio::{
    net::TcpStream,
    sync::{watch, Notify, OwnedSemaphorePermit, Semaphore},
    time::sleep,
};

use crate::config::HttpConfig;

const INITIAL_REQUEST_HEADER_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECTION_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_secs(1);
const HTTP1_HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP2_MAX_CONCURRENT_STREAMS: u32 = 100;
const HTTP2_MAX_HEADER_LIST_SIZE: u32 = 16 * 1024;

#[derive(Clone, Copy)]
struct Limits {
    max_connections: usize,
    initial_request_header_timeout: Duration,
    max_connection_age: Duration,
    drain_timeout: Duration,
    http1_header_read_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self::from(&HttpConfig::default())
    }
}

impl From<&HttpConfig> for Limits {
    fn from(config: &HttpConfig) -> Self {
        Self {
            max_connections: config.max_connections,
            initial_request_header_timeout: INITIAL_REQUEST_HEADER_TIMEOUT,
            max_connection_age: Duration::from_secs(config.max_connection_age_seconds),
            drain_timeout: CONNECTION_DRAIN_TIMEOUT,
            http1_header_read_timeout: HTTP1_HEADER_READ_TIMEOUT,
        }
    }
}

pub(crate) struct HttpServer {
    shutdown: watch::Sender<()>,
}

impl HttpServer {
    pub(crate) fn spawn(
        listener: TcpListener,
        app: Router,
        config: &HttpConfig,
    ) -> io::Result<Self> {
        Self::spawn_with_limits(listener, app, Limits::from(config))
    }

    fn spawn_with_limits(listener: TcpListener, app: Router, limits: Limits) -> io::Result<Self> {
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let (shutdown, shutdown_receiver) = watch::channel(());

        tokio::spawn(serve(listener, app, limits, shutdown_receiver));

        Ok(Self { shutdown })
    }

    pub(crate) fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }
}

async fn serve(
    listener: tokio::net::TcpListener,
    app: Router,
    limits: Limits,
    mut shutdown: watch::Receiver<()>,
) {
    let connection_slots = Arc::new(Semaphore::new(limits.max_connections));
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(limits.http1_header_read_timeout);
    builder
        .http2()
        .max_concurrent_streams(HTTP2_MAX_CONCURRENT_STREAMS)
        .max_header_list_size(HTTP2_MAX_HEADER_LIST_SIZE);
    let builder = Arc::new(builder);

    loop {
        let accepted = tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            accepted = listener.accept() => accepted,
        };

        let (stream, peer_address) = match accepted {
            Ok(connection) => connection,
            Err(error) => {
                tracing::warn!(%error, "failed to accept HTTP connection; retrying");
                tokio::select! {
                    biased;
                    _ = shutdown.changed() => return,
                    _ = sleep(ACCEPT_ERROR_BACKOFF) => {}
                }
                continue;
            }
        };

        let Ok(connection_slot) = Arc::clone(&connection_slots).try_acquire_owned() else {
            // Refuse excess connections immediately instead of allocating another task.
            drop(stream);
            continue;
        };

        tokio::spawn(serve_connection(
            Arc::clone(&builder),
            app.clone(),
            stream,
            peer_address,
            limits,
            shutdown.clone(),
            connection_slot,
        ));
    }
}

async fn serve_connection(
    builder: Arc<auto::Builder<TokioExecutor>>,
    app: Router,
    stream: TcpStream,
    peer_address: SocketAddr,
    limits: Limits,
    mut shutdown: watch::Receiver<()>,
    _connection_slot: OwnedSemaphorePermit,
) {
    let first_request_headers_received = Arc::new(Notify::new());
    let request_headers_notification = Arc::clone(&first_request_headers_received);
    // This middleware runs only after Hyper has parsed a complete request
    // header block, keeping the deadline armed through negotiation and parsing.
    let app = app
        .layer(Extension(ConnectInfo(peer_address)))
        .layer(map_request(move |request: Request| {
            request_headers_notification.notify_one();
            std::future::ready(request)
        }));
    let service = TowerToHyperService::new(app);
    let connection = builder.serve_connection(TokioIo::new(stream), service);
    let maximum_age = sleep(limits.max_connection_age);
    let first_request_headers = first_request_headers_received.notified();
    let initial_request_header_deadline = sleep(limits.initial_request_header_timeout);
    tokio::pin!(
        connection,
        maximum_age,
        first_request_headers,
        initial_request_header_deadline
    );
    let mut awaiting_first_request_headers = true;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            result = connection.as_mut() => {
                if let Err(error) = result {
                    tracing::debug!(%error, %peer_address, "HTTP connection closed with an error");
                }
                return;
            }
            _ = maximum_age.as_mut() => break,
            _ = first_request_headers.as_mut(), if awaiting_first_request_headers => {
                awaiting_first_request_headers = false;
            }
            _ = initial_request_header_deadline.as_mut(), if awaiting_first_request_headers => {
                tracing::debug!(%peer_address, "HTTP connection did not produce initial request headers in time");
                return;
            }
        }
    }

    // For HTTP/2 this sends GOAWAY and refuses new streams. HTTP/1 stops accepting
    // new requests on this connection. Already accepted requests may finish.
    connection.as_mut().graceful_shutdown();

    tokio::select! {
        biased;
        _ = shutdown.changed() => {
            tracing::debug!(%peer_address, "HTTP connection drain interrupted by server shutdown");
        }
        result = connection.as_mut() => {
            if let Err(error) = result {
                tracing::debug!(%error, %peer_address, "HTTP connection closed with an error while draining");
            }
        }
        _ = sleep(limits.drain_timeout) => {
            tracing::debug!(%peer_address, "HTTP connection exceeded its drain timeout");
        }
    }
}

#[cfg(test)]
mod tests;
