//! This example shows how to run a resolver node.

use std::num::NonZeroUsize;

use tracing::Level;
use tracing_subscriber;

use pkarr::{PkarrClient, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_env_filter("pkarr")
        .init();

    let client = PkarrClient::builder()
        .resolver()
        .cache_size(NonZeroUsize::new(1000).unwrap())
        .build()
        .unwrap();

    println!(
        "\nRunning a resolver at {} ...\n",
        client.local_addr().unwrap()
    );

    loop {}
}
