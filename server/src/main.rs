#[allow(dead_code)]
mod error;
mod server;

use anyhow::Result;
use std::num::NonZeroUsize;

use pkarr::{async_client::AsyncPkarrClient, PkarrClient};

use tracing::info;
use tracing_subscriber;

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

    let server = Server::spawn(client).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    server.shutdown().await?;

    Ok(())
}

struct Server {
    http_server: HttpServer,
}

impl Server {
    pub async fn spawn(client: AsyncPkarrClient) -> Result<Self> {
        let http_server = HttpServer::spawn(client).await?;
        Ok(Self { http_server })
    }

    /// Cancel the server tasks and wait for all tasks to complete.
    pub async fn shutdown(self) -> Result<()> {
        let _ = tokio::join!(self.http_server.shutdown());

        Ok(())
    }

    /// Wait for all tasks to complete.
    ///
    /// This will run forever unless all tasks close with an error, or `Self::cancel` is called.
    pub async fn run_until_error(self) -> Result<()> {
        tokio::select! {
            res = self.http_server.run_until_done() => res?,
        }
        Ok(())
    }
}
