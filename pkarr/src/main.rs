#![allow(unused)]
use crate::prelude::*;

use reqwest::blocking;
use url::Url;

mod error;
mod prelude;

#[derive(Debug)]
struct RelayClient {
    url: Url,
}

impl RelayClient {
    fn new(url: &str) -> Result<Self> {
        let _url = match Url::parse(url) {
            Ok(parsed) => parsed,
            Err(_) => return Err(Error::Static("Invalid relay URL")),
        };

        Ok(Self { url: _url })
    }

    fn get(self: RelayClient) -> Option<bytes::Bytes> {
        let mut url = self.url.to_owned();
        url.set_path("o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy");

        let response = match reqwest::blocking::get(url) {
            Ok(response) => response,
            Err(_) => return None,
        };

        if response.status() != reqwest::StatusCode::OK {
            return None;
        }

        let bytes = match response.bytes() {
            Ok(bytes) => bytes,
            Err(_) => return None,
        };

        Some(bytes)
    }
}

#[derive(Debug)]
struct Pkarr {
    // TODO: Use a single Client per [Pkarr] instance
    // TODO: add timeout to Client and test with a relay that does not end response stream.
    relays: Vec<RelayClient>,
}

impl Pkarr {
    fn new() -> Self {
        let default_relay_clients = vec!["https://relay.pkarr.org"];

        Self {
            relays: default_relay_clients
                .into_iter()
                .map(|url| RelayClient::new(url).unwrap())
                .collect(),
        }
    }

    fn resolve(self: Pkarr) -> Option<bytes::Bytes> {
        for relay in self.relays {
            match relay.get() {
                Some(bytes) => return Some(bytes),
                None => continue,
            }
        }

        None
    }
}

fn main() -> Result<()> {
    let client = Pkarr::new();

    dbg!(&client);

    let x = client.resolve();

    dbg!(x);

    Ok(())
}
