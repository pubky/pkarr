//! Make an HTTP request over to a Pkarr address using Reqwest

use std::sync::Arc;

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
async fn main() {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let cli = Cli::parse();
    let url = cli.url;

    let client = Client::builder().build().unwrap();

    let reqwest = reqwest::Client::builder()
        .dns_resolver(Arc::new(client))
        .build()
        .unwrap();

    let response = reqwest.request(Method::GET, &url).send().await.unwrap();

    let body = response.text().await.unwrap();

    println!("Resolved {url}\n{body}");
}
