//! This example shows how to publish [ResourceRecord]s directly to the DHT.
//!
//! Change the `Keypair::random()` to your own keypair to publish your own records.
//! Change the `packet.answers` to your own records.
//!
//! run this example from the project root:
//!     $ cargo run --example publish

use clap::Parser;
use std::time::Instant;
use tracing_subscriber;

use pkarr::{Client, Keypair, SignedPacket};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Publish to DHT only, Relays only, or default to both.
    mode: Option<Mode>,
}

#[derive(Debug, Clone)]
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

impl From<String> for Mode {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "dht" => Self::Dht,
            "relay" | "relays" => Self::Relays,
            _ => Self::Both,
        }
    }
}
