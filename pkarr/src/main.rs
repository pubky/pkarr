#![allow(unused)]
use crate::prelude::*;

use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use reqwest::blocking;
use std::time::SystemTime;
use url::Url;

pub use simple_dns::Packet;
pub use simple_dns::ResourceRecord;
pub use simple_dns::{rdata::*, Name, CLASS};
pub use std::net::Ipv4Addr;

mod error;
mod prelude;

#[derive(Debug, Clone)]
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

    fn get(self: RelayClient, id: &str) -> Result<Vec<u8>> {
        let mut url = self.url.to_owned();
        url.set_path(&id);

        let response = match reqwest::blocking::get(url) {
            Ok(response) => response,
            Err(err) => {
                dbg!(err);
                return Err(Error::Static("REQWUEST ERROR"));
            }
        };

        if response.status() != reqwest::StatusCode::OK {
            return Err(Error::Static("RELAY NON-OK STATUS"));
        }

        let bytes = match response.bytes() {
            Ok(bytes) => bytes,
            Err(err) => {
                dbg!(err);
                return Err(Error::Static("COULD NOT READ BYTES"));
            }
        };

        // let sig = bytes[0..64];
        // let seq = bytes[64..72];
        // let v = bytes[72..];
        let mut v: Vec<u8> = Vec::with_capacity(1000);
        v.extend_from_slice(&bytes[72..]);

        Ok(v)
    }

    fn put(self: &RelayClient, bep44_put_args: &Bep44PutArgs) -> Result<()> {
        let z32_key = zbase32::encode_full_bytes(&bep44_put_args.key);

        let mut url = self.url.to_owned();
        url.set_path(&z32_key);

        let client = reqwest::blocking::Client::new();

        let response = match client
            .put(url.to_owned())
            .body(bep44_put_args.clone())
            .send()
        {
            Ok(response) => response,
            Err(err) => {
                return Err(Error::Generic(format!("Relay PUT request failed {}", &url)));
            }
        };

        match response.status() {
            reqwest::StatusCode::OK => (),
            reqwest::StatusCode::BAD_REQUEST => {
                return Err(Error::Generic(format!(
                    "Relay PUT response: BAD_REQUEST {}",
                    &url
                )));
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
    fn new_current(signer: &SigningKey, value: Vec<u8>) -> Self {
        let sequence = system_time_now();
        let mut signable = format!("3:seqi{}e1:v{}:", sequence, value.len())
            .as_bytes()
            .to_vec();

        signable.extend(&value);

        dbg!(&signable);

        let signature = signer.sign(&signable);

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

fn relays_from_urls(urls: Vec<&str>) -> Vec<RelayClient> {
    urls.into_iter()
        .map(|url| RelayClient::new(url).unwrap())
        .collect()
}

impl Pkarr {
    fn new() -> Self {
        let default_relay_clients = vec!["https://relay.pkarr.org"];

        Self {
            relays: relays_from_urls(default_relay_clients),
        }
    }

    fn set_relays(self: &mut Pkarr, relays: Vec<&str>) {
        self.relays = relays_from_urls(relays)
    }

    fn resolve(self: Pkarr, key: VerifyingKey) -> Option<Vec<u8>> {
        let id = zbase32::encode_full_bytes(key.as_bytes());

        match self.relays.first().unwrap().clone().get(&id) {
            Ok(v) => {
                return Some(v);
            }
            Err(_) => {}
        }

        None
    }

    fn publish(self: &Pkarr, signer: &SigningKey, records: Vec<ResourceRecord>) -> Result<()> {
        let sequence = system_time_now();

        // Convert records into an encoded DNS Packet
        let mut packet = Packet::new_reply(0);
        packet.answers = records;
        let value = packet.build_bytes_vec_compressed()?;
        dbg!(&value, &value.len());

        let bep44_put_args = Bep44PutArgs::new_current(signer, value);

        // TODO: try publishing to the DHT directly if we have udp support. (requires DHT client)

        for relay in &self.relays {
            match relay.put(&bep44_put_args) {
                // Eagerly return success as long as one relay successfully publishes to the DHT.
                Ok(bytes) => return Ok(()),
                Err(err) => {
                    dbg!(err);
                    continue;
                }
            }
        }

        // All publishing attempts failed at this point.
        Err(Error::Static("Failed to publish"))
    }
}

fn main() -> Result<()> {
    let key = SigningKey::from([1; 32]);

    let mut alice = Pkarr::new();
    alice.set_relays(vec!["http://localhost:6881"]);

    let value = bytes::Bytes::from("[[\"foo\", \"zar\"]]");
    let x = alice.publish(
        &key,
        vec![
            ResourceRecord::new(
                Name::new("fff.foo").unwrap(),
                CLASS::IN,
                30,
                RData::A(A {
                    address: Ipv4Addr::new(127, 0, 0, 1).into(),
                }),
            ),
            ResourceRecord::new(
                Name::new("fff.foo").unwrap(),
                CLASS::IN,
                30,
                RData::A(A {
                    address: Ipv4Addr::new(127, 0, 0, 2).into(),
                }),
            ),
        ],
    );
    dbg!(x);

    let mut bob = Pkarr::new();
    bob.set_relays(vec!["http://localhost:6882"]);

    let z = bob.resolve(key.verifying_key()).unwrap();
    let y = Packet::parse(&z).unwrap();
    dbg!(&y);

    Ok(())
}
