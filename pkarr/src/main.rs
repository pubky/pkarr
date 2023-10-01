#![allow(unused)]
use crate::bep44::Bep44Args;
use crate::keys::{Keypair, PublicKey};
use crate::prelude::*;

use bytes::Bytes;
use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use reqwest::blocking;
use std::time::SystemTime;
use url::Url;

pub use simple_dns::Packet;
pub use simple_dns::ResourceRecord;
pub use simple_dns::{rdata::*, Name, CLASS};
pub use std::net::Ipv4Addr;

mod bep44;
mod error;
mod keys;
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

    fn get<'a>(self: RelayClient, public_key: &'a PublicKey) -> Result<Bep44Args> {
        let mut url = self.url.to_owned();
        url.set_path(&public_key.to_z32());

        let response = match reqwest::blocking::get(url) {
            Ok(response) => response,
            Err(err) => {
                dbg!(err);
                return Err(Error::Static("REQWUEST ERROR"));
            }
        };

        if response.status() != reqwest::StatusCode::OK {
            dbg!(response.status());
            return Err(Error::Static("RELAY NON-OK STATUS"));
        }

        let bytes = match response.bytes() {
            Ok(bytes) => bytes,
            Err(err) => {
                dbg!(err);
                return Err(Error::Static("COULD NOT READ BYTES"));
            }
        };

        Ok(Bep44Args::try_from_relay_response(&public_key, bytes)?)
    }

    fn put(self: &RelayClient, bep44args: &Bep44Args) -> Result<()> {
        let mut url = self.url.to_owned();
        url.set_path(&bep44args.k.to_z32());

        let client = reqwest::blocking::Client::new();

        let response = match client.put(url.to_owned()).body(bep44args).send() {
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
struct Pkarr {
    // TODO: Use a single Client per [Pkarr] instance
    // TODO: add timeout to Client and test with a relay that does not end response stream.
    relays: Vec<RelayClient>,
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

    fn resolve<'a>(self: Pkarr, public_key: &'a PublicKey) -> Option<Bep44Args> {
        match self.relays.first().unwrap().clone().get(public_key) {
            Ok(bep44args) => {
                dbg!(&bep44args);
                return Some(bep44args);
            }
            Err(err) => {
                dbg!(err);
            }
        }

        None
    }

    fn publish<'a>(self: &Pkarr, keypair: &Keypair, packet: &Packet<'a>) -> Result<()> {
        let bep44args = Bep44Args::try_from_packet(keypair, packet)?;

        let value = packet.build_bytes_vec_compressed()?;

        // let bep44_put_args = Bep44PutArgs::new_current(signer, value);

        // TODO: try publishing to the DHT directly if we have udp support. (requires DHT client)

        for relay in &self.relays {
            match relay.put(&bep44args) {
                // Eagerly return success as long as one relay successfully publishes to the DHT.
                Ok(bytes) => return Ok(()),
                Err(err) => {
                    dbg!(err);
                    continue;
                }
            }
        }

        // All publishing attempts failed at this point.
        Err(Error::PublishFailed)
    }
}

fn main() -> Result<()> {
    let keypair = Keypair::from_secret_key(&[0; 32]);

    // let mut alice = Pkarr::new();
    // alice.set_relays(vec!["http://localhost:6881"]);
    //
    // let mut packet = Packet::new_reply(0);
    // packet.answers.push(ResourceRecord::new(
    //     Name::new("_derp_region.iroh.").unwrap(),
    //     CLASS::IN,
    //     30,
    //     RData::A(A {
    //         address: Ipv4Addr::new(1, 1, 1, 1).into(),
    //     }),
    // ));
    //
    // let x = alice.publish(&keypair, &packet);
    //
    println!("\nResolving pk:{}...\n", keypair);
    let mut bob = Pkarr::new();
    bob.set_relays(vec!["http://localhost:6882"]);

    let public_key = keypair.public_key();

    let z = bob.resolve(&public_key);
    dbg!(&z.unwrap());

    Ok(())
}
