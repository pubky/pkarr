//! This example shows how to resolve [ResourceRecord]s through DHT, relays, or both.
//!
//! run this example from the project root:
//!     $ cargo run --example resolve <zbase32 encoded key>

use clap::{Parser, ValueEnum};
use pkarr::{Client, PublicKey, ResolvePolicy};
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Pkarr public key (z-base32 encoded) or a url where the TLD is a Pkarr key.
    public_key: String,
    /// Resolve from DHT only, Relays only, or default to both.
    #[arg(value_enum)]
    mode: Option<Mode>,
    /// List of relays (only valid if mode is 'relays')
    #[arg(requires = "mode")]
    relays: Option<Vec<String>>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Dht,
    Relays,
    Both,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("pkarr=info")
        .init();

    let cli = Cli::parse();

    let public_key: PublicKey = cli
        .public_key
        .as_str()
        .try_into()
        .expect("Invalid zbase32 encoded key");

    let mut builder = Client::builder();

    match cli.mode.unwrap_or(Mode::Both) {
        Mode::Dht => {
            builder.no_relays();
        }
        Mode::Relays => {
            builder.no_dht();

            if let Some(relays) = cli.relays {
                builder.relays(&relays).unwrap();
            }
        }
        _ => {}
    }

    let client = builder.build()?;

    println!("Resolving Pkarr: {} ...", cli.public_key);
    println!("\n=== COLD LOOKUP ===");
    resolve(&client, &public_key, ResolvePolicy::CacheFirst).await;

    println!("=== SUBSEQUENT LOOKUP ===");
    resolve(&client, &public_key, ResolvePolicy::CacheFirst).await;

    println!("Resolving most recent..");
    resolve(&client, &public_key, ResolvePolicy::NetworkOnly).await;

    Ok(())
}

async fn resolve(client: &Client, public_key: &PublicKey, policy: ResolvePolicy) {
    let start = Instant::now();

    match client.resolve(public_key, policy).await {
        Ok(signed_packet) => {
            println!(
                "\nResolved in {:?} milliseconds {}",
                start.elapsed().as_millis(),
                signed_packet
            );
        }
        Err(error) => {
            println!("\nFailed to resolve {}: {}", public_key, error);
        }
    }
}
