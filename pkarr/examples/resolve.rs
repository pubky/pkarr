//! This example shows how to resolve [ResourceRecord]s directly from the DHT.
//!
//! run this example from the project root:
//!     $ cargo run --example resolve <zbase32 encoded key>

use tracing::Level;
use tracing_subscriber;

use std::{
    thread::sleep,
    time::{Duration, Instant},
};

use clap::Parser;

use pkarr::PublicKey;

use pkarr::Client;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Mutable data public key.
    public_key: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_env_filter("pkarr")
        .init();

    let cli = Cli::parse();

    let public_key: PublicKey = cli
        .public_key
        .as_str()
        .try_into()
        .expect("Invalid zbase32 encoded key");

    let client = Client::builder().build()?;

    println!("Resolving Pkarr: {} ...", cli.public_key);
    println!("\n=== COLD LOOKUP ===");
    resolve(&client, &public_key).await;

    // loop {
    sleep(Duration::from_secs(1));
    println!("=== SUBSEQUENT LOOKUP ===");
    resolve(&client, &public_key).await;
    // }

    Ok(())
}

async fn resolve(client: &Client, public_key: &PublicKey) {
    let start = Instant::now();

    match client.resolve(public_key).await {
        Ok(Some(signed_packet)) => {
            println!(
                "\nResolved in {:?} milliseconds {}",
                start.elapsed().as_millis(),
                signed_packet
            );
        }
        Ok(None) => {
            println!("\nFailed to resolve {}", public_key);
        }
        Err(error) => {
            println!("Got error: {:?}", error)
        }
    }
}
