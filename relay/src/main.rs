use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

use pkarr_relay::Relay;

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
                .unwrap_or("pkarr_relay=info,tower_http=debug".to_string()),
        )
        .init();

    let relay = unsafe {
        if args.testnet {
            Relay::run_testnet().await?
        } else if let Some(config_path) = args.config {
            Relay::run_with_config_file(config_path).await?
        } else {
            Relay::builder().run().await?
        }
    };

    shutdown_signal().await?;

    info!("shutdown");

    relay.shutdown();

    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() -> tokio::io::Result<()> {
    use tokio::signal::unix::{signal, SignalKind};

    let mut terminate = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result,
        _ = terminate.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> tokio::io::Result<()> {
    tokio::signal::ctrl_c().await
}
