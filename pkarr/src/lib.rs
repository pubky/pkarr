#![allow(unused)]
use bytes::Bytes;
use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use reqwest::{Client, StatusCode};
use std::time::SystemTime;

// Rexports
pub use simple_dns as dns;
pub use url::Url;

// Modules

mod error;
mod keys;
mod signed_packet;

// Exports
pub use crate::error::Error;
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::SignedPacket;

// TODO: Make sure it is a reply packet
// TODO: test an unresponsive relay ()
// TODO: Add compare() method and compare result from relay to possibly cached result
// TODO: Add cache at all

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub const DEFAULT_PKARR_RELAY: &str = "https://relay.pkarr.org";

#[derive(Debug, Clone)]
pub struct RelayClient {
    url: Url,
}

impl RelayClient {
    pub fn new(url: &str) -> Result<Self> {
        let url = Url::parse(url).map_err(|_| Error::Static("Invalid relay URL"))?;
        Ok(Self { url })
    }

    pub async fn get(&self, public_key: PublicKey) -> Result<SignedPacket> {
        let mut url = self.url.clone();
        url.set_path(&public_key.to_z32());
        let response = Client::new().get(url.clone()).send().await.map_err(|err| {
            dbg!(err);
            Error::Static("REQWUEST ERROR")
        })?;
        if response.status() != StatusCode::OK {
            dbg!(response.status());
            return Err(Error::Static("RELAY NON-OK STATUS"));
        }
        let bytes = response.bytes().await.map_err(|err| {
            dbg!(err);
            Error::Static("COULD NOT READ BYTES")
        })?;
        Ok(SignedPacket::from_bytes(public_key, bytes)?)
    }

    pub async fn put(&self, signed_packet: SignedPacket) -> Result<()> {
        let mut url = self.url.to_owned();
        url.set_path(&signed_packet.public_key().to_z32());
        let response = Client::new()
            .put(url.clone())
            .body(Bytes::from(signed_packet))
            .send()
            .await
            .map_err(|err| {
                dbg!(err);
                Error::Generic(format!("Relay PUT request failed {}", &url))
            })?;
        if response.status() != StatusCode::OK {
            dbg!(response.status());
            return Err(Error::Static("RELAY NON-OK STATUS"));
        }
        Ok(())
    }
}

//
// #[derive(Debug)]
// struct Pkarr {
//     pub client: reqwest::blocking::Client,
//     // TODO: add timeout to Client and test with a relay that does not end response stream.
//     relays: Vec<Url>,
// }
//
// impl Pkarr {
//     fn new() -> Self {
//         let default_relay_clients = vec!["https://relay.pkarr.org"]
//             .into_iter()
//             .map(|url| Url::parse(url).unwrap())
//             .collect();
//
//         Self {
//             client: reqwest::blocking::Client::new(),
//             relays: default_relay_clients,
//         }
//     }
//
//     fn set_relays(self: &mut Pkarr, urls: Vec<&str>) -> Result<()> {
//         let relays: std::result::Result<Vec<_>, _> =
//             urls.into_iter().map(|url| Url::parse(url)).collect();
//
//         self.relays = match relays {
//             Ok(urls) => urls,
//             Err(err) => return Err(Error::Static("Invalid relay url")),
//         };
//
//         Ok(())
//     }
//
//     fn get_from_relay<'a>(&self, relay: &Url, public_key: &'a PublicKey) -> Result<SignedPacket> {
//         let mut url = relay.clone();
//         url.set_path(&public_key.to_z32());
//
//         let response = match self.client.get(url.to_owned()).send() {
//             Ok(response) => response,
//             Err(err) => {
//                 dbg!(err);
//                 return Err(Error::Static("REQWUEST ERROR"));
//             }
//         };
//
//         if response.status() != reqwest::StatusCode::OK {
//             dbg!(response.status());
//             return Err(Error::Static("RELAY NON-OK STATUS"));
//         }
//
//         let bytes = match response.bytes() {
//             Ok(bytes) => bytes,
//             Err(err) => {
//                 dbg!(err);
//                 return Err(Error::Static("COULD NOT READ BYTES"));
//             }
//         };
//
//         Ok(SignedPacket::try_from_relay_response(
//             public_key.clone(),
//             bytes,
//         )?)
//     }
//
//     pub fn resolve<'a>(&self, public_key: &'a PublicKey) -> Option<SignedPacket> {
//         for url in &self.relays {
//             match self.get_from_relay(url, public_key) {
//                 Ok(signed_packet) => return Some(signed_packet),
//                 Err(_) => {}
//             }
//         }
//
//         None
//     }
//
//     fn put_to_relay(&self, url: &Url, signed_packet: &SignedPacket) -> Result<()> {
//         let mut url = url.clone();
//         url.set_path(&signed_packet.public_key.to_z32());
//
//         let response = match self
//             .client
//             .put(url.to_owned())
//             .body(signed_packet.into_relay_payload())
//             .send()
//         {
//             Ok(response) => response,
//             Err(err) => {
//                 return Err(Error::Generic(format!("Relay PUT request failed {}", &url)));
//             }
//         };
//
//         match response.status() {
//             reqwest::StatusCode::OK => (),
//             reqwest::StatusCode::BAD_REQUEST => {
//                 return Err(Error::Generic(format!(
//                     "Relay PUT response: BAD_REQUEST {}",
//                     &url
//                 )));
//             }
//             _ => (),
//         }
//
//         Ok(())
//     }
//
//     pub fn publish<'a>(&self, signed_packet: &SignedPacket) -> Result<()> {
//         // TODO: try publishing to the DHT directly if we have udp support. (requires DHT client)
//
//         for url in &self.relays {
//             match self.put_to_relay(url, &signed_packet) {
//                 // Eagerly return success as long as one relay successfully publishes to the DHT.
//                 Ok(bytes) => return Ok(()),
//                 Err(err) => {
//                     dbg!(err);
//                     continue;
//                 }
//             }
//         }
//
//         // All publishing attempts failed at this point.
//         Err(Error::PublishFailed)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use std::net::Ipv4Addr;
//
//     use crate::PublicKey;
//
//     use super::{Keypair, PacketBuilder, Pkarr};
//
//     #[test]
//     fn resolve() {
//         let keypair = Keypair::random();
//
//         let mut alice = Pkarr::new();
//         alice.set_relays(vec!["http://localhost:6881"]);
//
//         let packet = PacketBuilder::new(&keypair)
//             .add_ip("_derp_region.iroh", Ipv4Addr::new(1, 1, 1, 1).into())
//             .build()
//             .unwrap();
//
//         let x = alice.publish(&packet);
//
//         println!("\nResolving pk:{}...\n", keypair);
//         let mut bob = Pkarr::new();
//         bob.set_relays(vec!["http://localhost:6882"]);
//
//         let public_key = keypair.public_key();
//
//         let packet = bob.resolve(&public_key).unwrap();
//         println!("{}", &packet);
//     }
// }
