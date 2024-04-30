mod cache;
mod error;
mod handlers;
mod rate_limiting;
mod server;

use anyhow::Result;
use cache::HeedPkarrCache;
use std::env;
use std::fs;
use std::path::Path;
use tracing::{info, Level};

use pkarr::PkarrClient;
use server::HttpServer;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_env_filter("pkarr=debug")
        .init();

    // TODO: config
    let exe_path = env::current_exe().expect("Failed to get current executable path");
    let dir_path = exe_path
        .parent()
        .expect("Failed to get directory of the current executable");
    let env_path = Path::new(dir_path).join("../storage/pkarr-server/pkarr-cache");
    fs::create_dir_all(&env_path)?;

    let cache = Box::new(HeedPkarrCache::new(&env_path, 1).unwrap());

    let client = PkarrClient::builder()
        .port(6881)
        .cache(cache)
        .build()
        .unwrap()
        .as_async();

    let http_server = HttpServer::spawn(client).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    http_server.shutdown().await?;

    Ok(())
}
