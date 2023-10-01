#![allow(unused)]
use simple_dns::Packet;
use simple_dns::ResourceRecord;
use simple_dns::{rdata::*, Name, CLASS};
use std::net::Ipv4Addr;

use url::Url;

mod error;
mod prelude;

mod keys;
mod packet;

pub use keys::{Keypair, PublicKey};
pub use packet::{PacketBuilder, SignedPacket};
use prelude::*;

// TODO: test an unresponsive relay (timeouts or use tokio?)
// TODO: Normalize names
// TODO: Add get record api

#[derive(Debug)]
struct Pkarr {
    client: reqwest::Client,
    // TODO: Use a single Client per [Pkarr] instance
    // TODO: add timeout to Client and test with a relay that does not end response stream.
    relays: Vec<Url>,
}

impl Pkarr {
    fn new() -> Self {
        let default_relay_clients = vec!["https://relay.pkarr.org"]
            .into_iter()
            .map(|url| Url::parse(url).unwrap())
            .collect();

        Self {
            client: reqwest::Client::new(),
            relays: default_relay_clients,
        }
    }

    fn set_relays(self: &mut Pkarr, urls: Vec<&str>) -> Result<()> {
        let relays: std::result::Result<Vec<_>, _> =
            urls.into_iter().map(|url| Url::parse(url)).collect();

        self.relays = match relays {
            Ok(urls) => urls,
            Err(err) => return Err(Error::Static("Invalid relay url")),
        };

        Ok(())
    }

    fn get_from_relay<'a>(&self, relay: &Url, public_key: &'a PublicKey) -> Result<SignedPacket> {
        let mut url = relay.clone();
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

        Ok(SignedPacket::try_from_relay_response(
            public_key.clone(),
            bytes,
        )?)
    }

    pub fn resolve<'a>(&self, public_key: &'a PublicKey) -> Option<SignedPacket> {
        for url in &self.relays {
            match self.get_from_relay(url, public_key) {
                Ok(signed_packet) => return Some(signed_packet),
                Err(_) => {}
            }
        }

        None
    }

    fn put_to_relay(&self, url: &Url, signed_packet: &SignedPacket) -> Result<()> {
        let mut url = url.clone();
        url.set_path(&signed_packet.public_key.to_z32());

        let client = reqwest::blocking::Client::new();

        let response = match client
            .put(url.to_owned())
            .body(signed_packet.into_relay_payload())
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

    pub fn publish<'a>(&self, signed_packet: &SignedPacket) -> Result<()> {
        // TODO: try publishing to the DHT directly if we have udp support. (requires DHT client)

        for url in &self.relays {
            match self.put_to_relay(url, &signed_packet) {
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

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use super::{Keypair, PacketBuilder, Pkarr};

    #[test]
    fn resolve() {
        let keypair = Keypair::from_secret_key(&[0; 32]);
        //
        // let mut alice = Pkarr::new();
        // alice.set_relays(vec!["http://localhost:6881"]);
        //
        // let packet = PacketBuilder::new(&keypair)
        //     .add_ip("_derp_region.iroh", Ipv4Addr::new(1, 1, 1, 1).into())
        //     .build()
        //     .unwrap();
        //
        // let x = alice.publish(&packet);

        println!("\nResolving pk:{}...\n", keypair);
        let mut bob = Pkarr::new();
        bob.set_relays(vec!["http://localhost:6882"]);

        let public_key = keypair.public_key();

        let z = bob.resolve(&public_key).unwrap();
        dbg!(&z);
    }
}
