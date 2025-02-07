//! This example shows how to resolve [ResourceRecord]s directly from the DHT.
//!
//! run this example from the project root:
//!     $ cargo run --example resolve <zbase32 encoded key>

use clap::{Parser, ValueEnum};
use std::time::Instant;
use tracing_subscriber;

use pkarr::{Client, PublicKey};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Mutable data public key.
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
    resolve(&client, &public_key, false).await;

    println!("=== SUBSEQUENT LOOKUP ===");
    resolve(&client, &public_key, false).await;

    println!("Resolving most recent..");
    resolve(&client, &public_key, true).await;

    Ok(())
}

async fn resolve(client: &Client, public_key: &PublicKey, most_recent: bool) {
    let start = Instant::now();

    match if most_recent {
        client.resolve_most_recent(public_key).await
    } else {
        client.resolve(public_key).await
    } {
        Some(signed_packet) => {
            println!(
                "\nResolved in {:?} milliseconds {}",
                start.elapsed().as_millis(),
                signed_packet
            );
        }
        None => {
            println!("\nFailed to resolve {}", public_key);
        }
    }
}
