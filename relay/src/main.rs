use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

use pkarr_relay::{Config, Relay};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to config file
    #[clap(short, long)]
    config: Option<PathBuf>,

    /// [tracing_subscriber::EnvFilter]
    #[clap(short, long)]
    tracing_env_filter: Option<String>,

    /// Run a Pkarr relay on a local testnet (port 15411).
    #[clap(long)]
    testnet: bool,
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

    let relay = unsafe {
        if args.testnet {
            Relay::start_testnet().await?
        } else if let Some(config_path) = args.config {
            Relay::start(Config::load(config_path).await?).await?
        } else {
            Relay::builder().start().await?
        }
    };

    tokio::signal::ctrl_c().await?;

    info!("shutdown");

    relay.shutdown();

    Ok(())
}
