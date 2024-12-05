use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::{debug, info};

use pkarr_server::{Config, Relay};

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

    let relay = unsafe { Relay::new(config).await? };

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    relay.shutdown();

    Ok(())
}
