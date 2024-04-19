use tracing::Level;
use tracing_subscriber;

use std::time::Instant;

use pkarr::{PkarrClient, PublicKey};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Mutable data public key.
    public_key: String,
}

fn main() {
    tracing_subscriber::fmt()
        // Switch to DEBUG to see incoming values and the IP of the responding nodes
        .with_max_level(Level::INFO)
        .init();

    let cli = Cli::parse();

    let public_key: PublicKey = cli
        .public_key
        .as_str()
        .try_into()
        .expect("Invalid zbase32 encoded key");

    let client = PkarrClient::builder().build().unwrap();

    println!("Resolving Pkarr: {} ...", cli.public_key);
    println!("\n=== COLD LOOKUP ===");
    resolve(&client, &public_key);

    println!("=== SUBSEQUENT LOOKUP ===");
    resolve(&client, &public_key);
}

fn resolve(client: &PkarrClient, public_key: &PublicKey) {
    let start = Instant::now();

    if let Ok(signed_packet) = client.resolve(public_key) {
        println!(
            "\nResolved in {:?} milliseconds {}",
            start.elapsed().as_millis(),
            signed_packet
        );
    } else {
        println!("\nFailed to resolve {}", public_key);
    }
}
