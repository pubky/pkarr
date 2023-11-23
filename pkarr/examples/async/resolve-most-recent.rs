//! This example shows how to resolve [ResourceRecord]s directly from the DHT.
//!
//! run this example from the project root:
//!     $ cargo run --example resolve <zbase32 encoded key>

use std::time::Instant;

use pkarr::{PkarrClient, PublicKey};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// z32 public_key
    public_key: String,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let client = PkarrClient::new();

    let str: &str = &cli.public_key;
    let public_key: PublicKey = str.try_into().expect("Invalid zbase32 encoded key");

    let instant = Instant::now();

    println!("\nResolving pk:{} ...", public_key);

    if let Some(signed_packet) = client.resolve_most_recent(public_key).await {
        println!("\nResolved in {:?} {}", instant.elapsed(), signed_packet);
    } else {
        println!("\nFailed to resolve {}", str);
    }
}
