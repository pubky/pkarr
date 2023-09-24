#![allow(unused)]
use crate::prelude::*;

use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use reqwest::blocking;
use std::time::SystemTime;
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

    fn put(self: RelayClient, bep44_put_args: &Bep44PutArgs) -> Result<()> {
        let z32_key = zbase32::encode_full_bytes(&bep44_put_args.key);

        let mut url = self.url.to_owned();
        url.set_path(&z32_key);

        let client = reqwest::blocking::Client::new();

        let response = match client.put(url).body(bep44_put_args.clone()).send() {
            Ok(response) => response,
            Err(err) => {
                dbg!(err);
                return Err(Error::Static("Relay PUT request failed"));
            }
        };

        match response.status() {
            reqwest::StatusCode::OK => (),
            reqwest::StatusCode::BAD_REQUEST => {
                return Err(Error::Static("Relay PUT response: BAD_REQUEST"));
            }
            _ => (),
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Bep44PutArgs {
    key: [u8; 32],
    sequence: u64,
    value: Vec<u8>,
    /// The signable is the string `3:seqi{sequence}e1:v{value.len()}:{value}`
    signable: Vec<u8>,
    signature: [u8; 64],
}

impl Bep44PutArgs {
    fn new_current(signer: SigningKey, value: Vec<u8>) -> Self {
        let sequence = system_time_now();
        let mut signable = format!(
            "3:seqi{}e1:v{}:{}",
            sequence,
            value.len(),
            String::from_utf8(value.clone()).unwrap()
        );

        let signature = signer.sign(signable.as_bytes());

        Self {
            key: *signer.verifying_key().as_bytes(),
            sequence,
            value,
            signable: signable.into(),
            signature: signature.to_bytes(),
        }
    }
}

impl Clone for Bep44PutArgs {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            sequence: self.sequence,
            value: self.value.clone(),
            signable: self.signable.clone(),
            signature: self.signature,
        }
    }
}

impl From<Bep44PutArgs> for reqwest::blocking::Body {
    fn from(bep44_put_args: Bep44PutArgs) -> reqwest::blocking::Body {
        let mut body = Vec::with_capacity(64 + 8 + bep44_put_args.value.len());

        body.extend_from_slice(&bep44_put_args.signature);
        body.extend_from_slice(&bep44_put_args.sequence.to_be_bytes());
        body.extend_from_slice(&bep44_put_args.value);

        reqwest::blocking::Body::from(body)
    }
}

#[derive(Debug)]
struct Pkarr {
    // TODO: Use a single Client per [Pkarr] instance
    // TODO: add timeout to Client and test with a relay that does not end response stream.
    relays: Vec<RelayClient>,
}

fn system_time_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time drift")
        .as_micros() as u64
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

    fn publish(self: Pkarr, signer: SigningKey) -> Result<()> {
        let sequence = system_time_now();
        let value = bytes::Bytes::from("Hello world!");

        let bep44_put_args = Bep44PutArgs::new_current(signer, value.into());

        // TODO: try publishing to the DHT directly if we have udp support. (requires DHT client)

        for relay in self.relays {
            match relay.put(&bep44_put_args) {
                // Eagerly return success as long as one relay successfully publishes to the DHT.
                Ok(bytes) => return Ok(()),
                Err(err) => continue,
            }
        }

        // All publishing attempts failed at this point.
        Err(Error::Static("Failed to publish"))
    }
}

fn main() -> Result<()> {
    let client = Pkarr::new();

    let x = client.publish(SigningKey::from([0; 32]));

    dbg!(x);

    Ok(())
}
