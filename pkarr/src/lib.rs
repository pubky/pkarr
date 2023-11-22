#![doc = include_str!("./README.md")]

use bytes::Bytes;
use mainline::Dht;
use url::Url;

// Rexports
pub use bytes;
pub use simple_dns as dns;
pub use url;

// Modules

mod error;
mod keys;
mod signed_packet;

// Exports
pub use crate::error::Error;
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::SignedPacket;

// TODO: Make sure it is a reply packet
// TODO: Add compare() method to SignedPacket to compare to cached packets.
//
// TODO: Add Cache of previously seen packets to compare new results to it.
// TODO: Implement `get` and `put` methods to concurrently query multiple relays.
// TODO: Add `thorough_get` that queries all relays (and DHT nodes) until it sees a _new_ packet or
// exauhsts all sources.

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub const DEFAULT_PKARR_RELAY: &str = "https://relay.pkarr.org";

#[derive(Debug, Clone)]
/// Main client for publishing and resolving [SginedPacket](crate::SignedPacket)s.
pub struct PkarrClient {
    http_client: reqwest::Client,
    dht: Dht,
}

impl PkarrClient {
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            dht: Dht::default(),
        }
    }

    /// Resolves a [SignedPacket](crate::SignedPacket) from a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub async fn relay_get(&self, url: &Url, public_key: PublicKey) -> Result<SignedPacket> {
        let url = format_relay_url(url, &public_key);

        let response = self.http_client.get(url).send().await?;
        if !response.status().is_success() {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text().await?,
            ));
        }
        let bytes = response.bytes().await?;

        SignedPacket::from_bytes(public_key, bytes)
    }

    /// Publishes a [SignedPacket](crate::SignedPacket) through a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub async fn relay_put(&self, url: &Url, signed_packet: SignedPacket) -> Result<()> {
        let url = format_relay_url(url, signed_packet.public_key());

        let response = self
            .http_client
            .put(url)
            .body(Bytes::from(signed_packet))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text().await?,
            ));
        }

        Ok(())
    }

    /// Resolve the first resolved [SignedPacket](crate::SignedPacket) from the DHT.
    pub fn resolve(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.0, None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                return Some(signed_packet);
            };
        }

        None
    }
}

impl Default for PkarrClient {
    fn default() -> Self {
        Self::new()
    }
}

fn format_relay_url(url: &Url, public_key: &PublicKey) -> Url {
    let mut url = url.to_owned();
    url.set_path(&public_key.to_z32());

    url
}
