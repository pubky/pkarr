use clap::{Parser, Subcommand, ValueEnum};
use pkarr::{Client, PublicKey};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pkarr-resolver")]
// TODO: Update description when more commands are added.
#[command(about = "Resolve PKARR records via CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Resolve a Pkarrr record by public key
    Resolve {
        /// Pkarr public key (z-base32 encoded) or a url where the TLD is a Pkarr key.
        public_key: String,
        /// Resolve from DHT only, Relays only, or default to both.
        #[arg(value_enum)]
        mode: Option<Mode>,
        /// List of relays (only valid if mode is 'relays')
        #[arg(requires = "mode")]
        relays: Option<Vec<String>>,
    },
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

    match &cli.command {
        Commands::Resolve {
            public_key,
            mode,
            relays,
        } => {
            let public_key: PublicKey = public_key
                .as_str()
                .try_into()
                .expect("Invalid zbase32 encoded key");

            let mut builder = Client::builder();

            match mode.clone().unwrap_or(Mode::Both) {
                Mode::Dht => {
                    builder.no_relays();
                }
                Mode::Relays => {
                    builder.no_dht();

                    if let Some(relays) = relays {
                        builder.relays(relays).unwrap();
                    }
                }
                _ => {}
            }

            let client = builder.build()?;

            println!("Resolving Pkarr: {public_key} ...");
            let start = Instant::now();
            match client.resolve(&public_key).await {
                Some(signed_packet) => {
                    println!(
                        "\nResolved in {:?} milliseconds {}",
                        start.elapsed().as_millis(),
                        signed_packet
                    );

                    println!("Checking if there are more recent recent packets..");
                    let start = Instant::now();
                    if let Some(more_recent_maybe) = client.resolve_most_recent(&public_key).await {
                        if more_recent_maybe.more_recent_than(&signed_packet) {
                            println!(
                                "\nResolved a more recent packet in {:?} milliseconds {}",
                                start.elapsed().as_millis(),
                                signed_packet
                            );

                            return Ok(());
                        }
                    }

                    println!(".. no more recent packets.");
                }
                None => {
                    println!("\nFailed to resolve {public_key}");
                }
            };

            Ok(())
        }
    }
}
