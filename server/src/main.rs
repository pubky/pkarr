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
use pkarr::{mainline::dht::DhtSettings, Client};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to config file
    #[clap(short, long)]
    config: Option<PathBuf>,
    /// [tracing_subscriber::EnvFilter]
    #[clap(short, long)]
    tracing_env_filter: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            args.tracing_env_filter
                .unwrap_or("pkarr=info,tower_http=debug".to_string()),
        )
        .init();

    let config = if let Some(path) = args.config {
        Config::load(path).await?
    } else {
        Config::default()
    };

    debug!(?config, "Pkarr server config");

    let env_path = &config.cache_path()?;
    fs::create_dir_all(env_path)?;
    let cache = Box::new(HeedPkarrCache::new(env_path, config.cache_size()).unwrap());

    let rate_limiter = rate_limiting::IpRateLimiter::new(config.rate_limiter());

    let client = Client::builder()
        .dht_settings(DhtSettings {
            port: Some(config.dht_port()),
            server: Some(Box::new(dht_server::DhtServer::new(
                cache.clone(),
                config.resolvers(),
                config.minimum_ttl(),
                config.maximum_ttl(),
                rate_limiter.clone(),
            ))),
            ..DhtSettings::default()
        })
        .resolvers(config.resolvers())
        .minimum_ttl(config.minimum_ttl())
        .maximum_ttl(config.maximum_ttl())
        .cache(cache)
        .build()?
        .as_async();

    let udp_address = client.local_addr().unwrap();

    info!("Running as a resolver on UDP socket {udp_address}");

    let http_server = HttpServer::spawn(client, config.relay_port(), rate_limiter).await?;

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    http_server.shutdown().await?;

    Ok(())
}
