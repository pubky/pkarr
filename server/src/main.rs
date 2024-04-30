mod cache;
mod config;
mod error;
mod handlers;
mod rate_limiting;
mod server;

use anyhow::Result;
use cache::HeedPkarrCache;
use clap::Parser;
use config::Config;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, Level};

use pkarr::PkarrClient;
use server::HttpServer;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to config file
    #[clap(short, long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_env_filter("pkarr=debug")
        .init();

    // Config::load();
    let args = Cli::parse();

    let config = if let Some(path) = args.config {
        Config::load(path).await?
    } else {
        Config::default()
    };

    debug!(?config);

    let env_path = &config.pkarr_cache_path()?;
    fs::create_dir_all(env_path)?;
    let cache = Box::new(HeedPkarrCache::new(&env_path, 1_000_000).unwrap());

    let client = PkarrClient::builder()
        .port(config.dht_port())
        .resolver()
        .cache(cache)
        .build()
        .unwrap()
        .as_async();

    let http_server = HttpServer::spawn(client, config.relay_port()).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    http_server.shutdown().await?;

    Ok(())
}
