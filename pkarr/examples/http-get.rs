//! Make an HTTP request over to a Pkarr address using Reqwest

use reqwest::Method;
use tracing::Level;
use tracing_subscriber;

use clap::Parser;

use pkarr::Client;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Url to GET from
    url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let cli = Cli::parse();
    let url = cli.url;

    let client = Client::builder().build()?;

    let reqwest = reqwest::ClientBuilder::from(client).build()?;

    println!("GET {url}..");
    let response = reqwest.request(Method::GET, &url).send().await?;

    let body = response.text().await?;

    println!("{body}");

    Ok(())
}
