//! Public-Key Addressable Resource Records

// Rexports
pub use bytes::Bytes;
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
pub struct PkarrClient {
    http_client: reqwest::Client,
}

impl PkarrClient {
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
        }
    }

    pub async fn relay_get(&self, url: &Url, public_key: PublicKey) -> Result<SignedPacket> {
        let url = format_relay_url(url, &public_key);

        let response = self.http_client.get(url).send().await?;
        let bytes = response.bytes().await?;

        Ok(SignedPacket::from_bytes(public_key, bytes)?)
    }

    pub async fn relay_put(&self, url: &Url, signed_packet: SignedPacket) -> Result<()> {
        let url = format_relay_url(url, signed_packet.public_key());

        let response = self
            .http_client
            .put(url.clone())
            .body(Bytes::from(signed_packet))
            .send()
            .await?;

        if response.status() != reqwest::StatusCode::OK {
            return Err(Error::RelayResponse(
                url,
                response.status(),
                response.text().await?,
            ));
        }

        Ok(())
    }
}

fn format_relay_url(url: &Url, public_key: &PublicKey) -> Url {
    let mut url = url.to_owned();
    url.set_path(&public_key.to_z32());

    url
}
