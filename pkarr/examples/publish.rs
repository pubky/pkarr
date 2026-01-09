//! This example shows how to publish [ResourceRecord]s directly to the DHT.
//!
//! Change the `Keypair::random()` to your own keypair to publish your own records.
//! Change the `packet.answers` to your own records.
//!
//! run this example from the project root:
//!     $ cargo run --example publish

use clap::{Parser, ValueEnum};
use pkarr::{Client, Keypair, SignedPacket};
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Publish to DHT only, Relays only, or default to both.
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

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("_foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)?;

    let instant = Instant::now();

    println!("\nPublishing {} ...", keypair.public_key());

    match client.publish(&signed_packet, None).await {
        Ok(()) => {
            println!(
                "\nSuccessfully published {} in {:?}",
                keypair.public_key(),
                instant.elapsed(),
            );
        }
        Err(err) => {
            println!("\nFailed to publish {} \n {}", keypair.public_key(), err);
        }
    };

    Ok(())
}
