#![doc = include_str!("../README.md")]

#[cfg(feature = "dht")]
use mainline::{Dht, DhtSettings, GetMutableResponse, MutableItem, Response, StoreQueryMetdata};
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

pub struct PkarrClientBuilder {
    settings: DhtSettings,
}

impl PkarrClientBuilder {
    pub fn new() -> Self {
        Self {
            settings: DhtSettings::default(),
        }
    }

    pub fn bootstrap(mut self, bootstrap: &[String]) -> Self {
        self.settings.bootstrap = Some(bootstrap.to_owned());
        self
    }

    pub fn build(self) -> PkarrClient {
        PkarrClient {
            #[cfg(all(feature = "relay", not(feature = "async")))]
            http_client: reqwest::blocking::Client::new(),
            #[cfg(all(feature = "relay", feature = "async"))]
            http_client: reqwest::Client::new(),
            dht: Dht::new(self.settings),
        }
    }
}

impl Default for PkarrClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
/// Main client for publishing and resolving [SginedPacket](crate::SignedPacket)s.
pub struct PkarrClient {
    #[cfg(all(feature = "relay", not(feature = "async")))]
    http_client: reqwest::blocking::Client,
    #[cfg(all(feature = "relay", feature = "async"))]
    http_client: reqwest::Client,
    dht: Dht,
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

    pub fn builder() -> PkarrClientBuilder {
        PkarrClientBuilder::default()
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
    pub fn relay_get(&self, url: &Url, public_key: PublicKey) -> Result<Option<SignedPacket>> {
        let url = format_relay_url(url, &public_key);

        let response = self.http_client.get(url).send()?;

        if response.status().is_success() {
            let bytes = response.bytes()?;
            return Ok(Some(SignedPacket::from_relay_response(public_key, bytes)?));
        } else if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        } else {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text()?,
            ));
        }
    }

    #[cfg(all(feature = "relay", feature = "async"))]
    /// Resolves a [SignedPacket] from a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub async fn relay_get(
        &self,
        url: &Url,
        public_key: PublicKey,
    ) -> Result<Option<SignedPacket>> {
        let url = format_relay_url(url, &public_key);

        let response = self.http_client.get(url).send().await?;

        if response.status().is_success() {
            let bytes = response.bytes().await?;
            return Ok(Some(SignedPacket::from_relay_response(public_key, bytes)?));
        } else if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        } else {
            return Err(Error::RelayResponse(
                response.url().clone(),
                response.status(),
                response.text().await?,
            ));
        }
    }

    #[cfg(all(feature = "relay", not(feature = "async")))]
    /// Publishes a [SignedPacket] through a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub fn relay_put(&self, url: &Url, signed_packet: &SignedPacket) -> Result<()> {
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
    /// Publishes a [SignedPacket] through a [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
    pub async fn relay_put(&self, url: &Url, signed_packet: &SignedPacket) -> Result<()> {
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
    /// Publish a [SignedPacket] to the DHT.
    ///
    /// It performs a thorough lookup first to find the closest nodes,
    /// before storing the signed packet to them, so it may take few seconds.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<StoreQueryMetdata> {
        let item: MutableItem = signed_packet.into();
        self.dht.put_mutable(item).map_err(Error::MainlineError)
    }

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Eagerly return the first resolved [SignedPacket] from the DHT.
    pub fn resolve(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                return Some(signed_packet);
            };
        }

        None
    }

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Eagerly return the first resolved [SignedPacket] from the DHT.
    pub async fn resolve(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        for res in &mut response {
            let signed_packet: Result<SignedPacket> = res.item.try_into();
            if let Ok(signed_packet) = signed_packet {
                return Some(signed_packet);
            };
        }

        None
    }

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Fully traverse the DHT and the return the most recent resolved [SignedPacket].
    pub fn resolve_most_recent(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        let mut most_recent: Option<SignedPacket> = None;

        for next in &mut response {
            let next_packet: Result<SignedPacket> = next.item.try_into();
            if let Ok(next_packet) = next_packet {
                if let Some(most_recent) = &most_recent {
                    if most_recent.more_recent_than(&next_packet) {
                        continue;
                    }
                }

                most_recent = Some(next_packet)
            };
        }

        most_recent
    }

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Fully traverse the DHT and the return the most recent resolved [SignedPacket].
    pub async fn resolve_most_recent(&self, public_key: PublicKey) -> Option<SignedPacket> {
        let mut response = self.dht.get_mutable(public_key.as_bytes(), None);

        let mut most_recent: Option<SignedPacket> = None;

        for next in &mut response {
            let next_packet: Result<SignedPacket> = next.item.try_into();
            if let Ok(next_packet) = next_packet {
                if let Some(most_recent) = &most_recent {
                    if most_recent.more_recent_than(&next_packet) {
                        continue;
                    }
                }

                most_recent = Some(next_packet)
            };
        }

        most_recent
    }

    #[cfg(all(feature = "dht", not(feature = "async")))]
    /// Return `mainline's` [Response] to have the most control over the response and access its metadata.
    ///
    /// Mostly useful to terminate as soon as you find a [SignedPacket] that satisfies a specific
    /// condition, for example:
    /// - it is [more_recent_than](SignedPacket::more_recent_than) a cached one.
    /// - or it has a [timestamp](SignedPacket::timestamp) higher than a specific value.
    /// - or it contains specific [fresh_resource_records](SignedPacket::fresh_resource_records).
    ///
    /// Most likely you want to use [resolve](PkarrClient::resolve) or
    /// [resolve_most_recent](PkarrClient::resolve_most_recent) instead.
    pub fn resolve_raw(&self, public_key: PublicKey) -> Response<GetMutableResponse> {
        self.dht.get_mutable(public_key.as_bytes(), None)
    }

    #[cfg(all(feature = "dht", feature = "async"))]
    /// Return `mainline's` [Response] to have the most control over the response and access its metadata.
    ///
    /// Mostly useful to terminate as soon as you find a [SignedPacket] that satisfies a specific
    /// condition, for example:
    /// - it is [more_recent_than](SignedPacket::more_recent_than) a cached one.
    /// - or it has a [timestamp](SignedPacket::timestamp) higher than a specific value.
    /// - or it contains specific [fresh_resource_records](SignedPacket::fresh_resource_records).
    ///
    /// Most likely you want to use [resolve](PkarrClient::resolve) or
    /// [resolve_most_recent](PkarrClient::resolve_most_recent) instead.
    pub async fn resolve_raw(&self, public_key: PublicKey) -> Response<GetMutableResponse> {
        self.dht.get_mutable(public_key.as_bytes(), None)
    }
}

impl Default for PkarrClient {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[cfg(feature = "relay")]
fn format_relay_url(url: &Url, public_key: &PublicKey) -> Url {
    let mut url = url.to_owned();
    url.set_path(&public_key.to_z32());

    url
}
