mod cache;
mod config;
mod dht_server;
mod error;
mod handlers;
mod http_server;
mod rate_limiting;

use anyhow::Result;
use cache::HeedPkarrCache;
use clap::Parser;
use config::Config;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

use http_server::HttpServer;
use pkarr::{client::mainline::dht::DhtSettings, PkarrClient};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to config file
    #[clap(short, long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("pkarr=info,tower_http=debug")
        .init();

    // Config::load();
    let args = Cli::parse();

    let config = if let Some(path) = args.config {
        Config::load(path).await?
    } else {
        Config::default()
    };

    debug!(?config, "Pkarr server config");

    let env_path = &config.cache_path()?;
    fs::create_dir_all(env_path)?;
    let cache = Box::new(HeedPkarrCache::new(env_path, config.cache_size()).unwrap());

    let rate_limiter_layer = rate_limiting::create(config.rate_limiter());

    let client = PkarrClient::builder()
        .dht_settings(DhtSettings {
            port: Some(config.dht_port()),
            server: Some(Box::new(dht_server::DhtServer::new(
                cache.clone(),
                config.resolvers(),
                config.minimum_ttl(),
                config.maximum_ttl(),
                rate_limiter_layer.clone(),
            ))),
            ..DhtSettings::default()
        })
        .cache(cache)
        .build()?
        .as_async();

    let udp_address = client.local_addr().unwrap();

    info!("Running as a resolver on UDP socket {udp_address}");

    let http_server = HttpServer::spawn(client, config.relay_port(), rate_limiter_layer).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    http_server.shutdown().await?;

    Ok(())
}
