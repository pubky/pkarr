#[allow(dead_code)]
mod error;
mod handlers;
mod rate_limiting;
mod server;

use anyhow::Result;
use std::num::NonZeroUsize;
use tracing::info;

use pkarr::PkarrClient;
use server::HttpServer;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let client = PkarrClient::builder()
        .cache_size(NonZeroUsize::new(1_000_000).unwrap())
        // TODO: config
        .port(6881)
        .build()
        .unwrap()
        .as_async();

    let http_server = HttpServer::spawn(client).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    http_server.shutdown().await?;

    Ok(())
}
