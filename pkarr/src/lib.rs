#![doc = include_str!("./README.md")]

#[cfg(feature = "dht")]
use mainline::{
    common::{MutableItem, StoreQueryMetdata},
    Dht,
};
#[cfg(feature = "relay")]
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

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub const DEFAULT_PKARR_RELAY: &str = "https://relay.pkarr.org";

#[derive(Debug, Clone)]
/// Main client for publishing and resolving [SginedPacket](crate::SignedPacket)s.
pub struct PkarrClient {
    #[cfg(all(feature = "relay", not(feature = "async")))]
    http_client: reqwest::blocking::Client,
    #[cfg(all(feature = "relay", feature = "async"))]
    http_client: reqwest::Client,
    pub dht: Dht,
}

impl PkarrClient {
    pub fn new() -> Self {
        Self {
            #[cfg(all(feature = "relay", not(feature = "async")))]
            http_client: reqwest::blocking::Client::new(),
            #[cfg(all(feature = "relay", feature = "async"))]
            http_client: reqwest::Client::new(),
            dht: Dht::default(),
        }
    }

    #[cfg(all(feature = "relay", not(feature = "async")))]
    /// Add your own Reqwest blocking client with custom config.
    pub fn with_http_client(mut self, client: reqwest::blocking::Client) -> Self {
        self.http_client = client;
        self
    }

    #[cfg(all(feature = "relay", feature = "async"))]
    /// Add your own Reqwest client with custom config.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }

    #[cfg(all(feature = "relay", not(feature = "async")))]
    /// Resolves a [SignedPacket](crate::SignedPacket) from a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub fn relay_get(&self, url: &Url, public_key: PublicKey) -> Result<SignedPacket> {
        let url = format_relay_url(url, &public_key);

        let response = self.http_client.get(url).send()?;
        if !response.status().is_success() {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text()?,
            ));
        }
        let bytes = response.bytes()?;

        SignedPacket::from_relay_response(public_key, bytes)
    }

    #[cfg(all(feature = "relay", feature = "async"))]
    /// Async version of [relay_get](PkarrClient::relay_get)
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

        SignedPacket::from_relay_response(public_key, bytes)
    }

    #[cfg(all(feature = "relay", not(feature = "async")))]
    /// Publishes a [SignedPacket](crate::SignedPacket) through a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub fn relay_put(&self, url: &Url, signed_packet: SignedPacket) -> Result<()> {
        let url = format_relay_url(url, signed_packet.public_key());

        let response = self
            .http_client
            .put(url)
            .body(signed_packet.as_relay_request())
            .send()?;

        if !response.status().is_success() {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text()?,
            ));
        }

        Ok(())
    }

    #[cfg(all(feature = "relay", feature = "async"))]
    /// Async version of [relay_put](PkarrClient::relay_put)
    pub async fn relay_put(&self, url: &Url, signed_packet: SignedPacket) -> Result<()> {
        let url = format_relay_url(url, signed_packet.public_key());

        let response = self
            .http_client
            .put(url)
            .body(signed_packet.as_relay_request())
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

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Publish a [SignedPacket] to the DHT.
    ///
    /// It performs a thorough lookup first to find the closest nodes,
    /// before storing the signed packet to them, so it may take few seconds.
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<StoreQueryMetdata> {
        let item: MutableItem = signed_packet.into();
        self.dht.put_mutable(item).map_err(Error::MainlineError)
    }

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Async version of [publish](PkarrClient::publish)
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<StoreQueryMetdata> {
        let item: MutableItem = signed_packet.into();
        self.dht.put_mutable(item).map_err(Error::MainlineError)
    }

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Returns the first resolved [SignedPacket] from the DHT.
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

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Returns the first resolved [SignedPacket](crate::SignedPacket) from the DHT.
    pub async fn resolve(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.0, None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                return Some(signed_packet);
            };
        }

        None
    }

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Returns the most recent [SignedPacket] from the DHT.
    /// In order to determine the most recent, it has to do a full lookup first, so
    /// this method may take few seconds.
    pub fn resolve_most_recent(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.0, None);

        let mut most_recent: Option<SignedPacket> = None;

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                if let Some(most_recent) = &most_recent {
                    if signed_packet.timestamp() < most_recent.timestamp() {
                        continue;
                    }

                    // In the rare ocasion of timestamp collission,
                    // we use the one with the largest value
                    if signed_packet.timestamp() == most_recent.timestamp()
                        && signed_packet.encoded_packet() < most_recent.encoded_packet()
                    {
                        continue;
                    }
                }

                most_recent = Some(signed_packet)
            };
        }

        most_recent
    }

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Async version of [resolve_most_recent](PkarrClient::resolve_most_recent).
    pub async fn resolve_most_recent(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.0, None);

        let mut most_recent: Option<SignedPacket> = None;

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                if let Some(most_recent) = &most_recent {
                    if signed_packet.timestamp() < most_recent.timestamp() {
                        continue;
                    }

                    // In the rare ocasion of timestamp collission,
                    // we use the one with the largest value
                    if signed_packet.timestamp() == most_recent.timestamp() {
                        if signed_packet.encoded_packet() < most_recent.encoded_packet() {
                            continue;
                        }
                    }
                }

                most_recent = Some(signed_packet)
            };
        }

        most_recent
    }
}

impl Default for PkarrClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "relay")]
fn format_relay_url(url: &Url, public_key: &PublicKey) -> Url {
    let mut url = url.to_owned();
    url.set_path(&public_key.to_z32());

    url
}
