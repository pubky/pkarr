//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use std::num::NonZeroUsize;

use reqwest::{Response, StatusCode};
use tracing::debug;

#[cfg(target_arch = "wasm32")]
use futures::future::select_ok;

#[cfg(not(target_arch = "wasm32"))]
use tokio::task::JoinSet;

use crate::{
    Cache, Error, InMemoryCache, PublicKey, Result, SignedPacket, DEFAULT_CACHE_SIZE,
    DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL, DEFAULT_RELAYS,
};

#[derive(Debug, Clone)]
/// [Client]'s settings
pub struct Settings {
    pub relays: Vec<String>,
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: NonZeroUsize,
    /// Used in the `min` parameter in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MINIMUM_TTL]
    pub minimum_ttl: u32,
    /// Used in the `max` parametere in [SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MAXIMUM_TTL]
    pub maximum_ttl: u32,
    /// Custom [reqwest::Client]
    pub http_client: reqwest::Client,
    /// Custom [Cache] implementation, defaults to [InMemoryCache]
    pub cache: Option<Box<dyn Cache>>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            relays: DEFAULT_RELAYS.map(|s| s.into()).to_vec(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            http_client: reqwest::Client::new(),
            cache: None,
        }
    }
}

#[derive(Debug, Default)]
/// Builder for [Client]
pub struct ClientBuilder {
    settings: Settings,
}

impl ClientBuilder {
    /// Set the relays to publish and resolve [SignedPacket]s to and from.
    pub fn relays(mut self, relays: Vec<String>) -> Self {
        self.settings.relays = relays;
        self
    }

    /// Set the [Settings::cache_size].
    ///
    /// Controls the capacity of [Cache].
    pub fn cache_size(mut self, cache_size: NonZeroUsize) -> Self {
        self.settings.cache_size = cache_size;
        self
    }

    /// Set the [Settings::minimum_ttl] value.
    ///
    /// Limits how soon a [SignedPacket] is considered expired.
    pub fn minimum_ttl(mut self, ttl: u32) -> Self {
        self.settings.minimum_ttl = ttl;
        self.settings.maximum_ttl = self.settings.maximum_ttl.clamp(ttl, u32::MAX);
        self
    }

    /// Set the [Settings::maximum_ttl] value.
    ///
    /// Limits how long it takes before a [SignedPacket] is considered expired.
    pub fn maximum_ttl(mut self, ttl: u32) -> Self {
        self.settings.maximum_ttl = ttl;
        self.settings.minimum_ttl = self.settings.minimum_ttl.clamp(0, ttl);
        self
    }

    /// Set a custom implementation of [Cache].
    pub fn cache(mut self, cache: Box<dyn Cache>) -> Self {
        self.settings.cache = Some(cache);
        self
    }

    pub fn build(self) -> Result<Client> {
        Client::new(self.settings)
    }
}

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct Client {
    http_client: reqwest::Client,
    relays: Vec<String>,
    cache: Box<dyn Cache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl Default for Client {
    fn default() -> Self {
        Self::new(Settings::default()).unwrap()
    }
}

impl Client {
    pub fn new(settings: Settings) -> Result<Self> {
        if settings.relays.is_empty() {
            return Err(Error::EmptyListOfRelays);
        }

        let cache = settings
            .cache
            .clone()
            .unwrap_or(Box::new(InMemoryCache::new(settings.cache_size)));

        Ok(Self {
            http_client: settings.http_client,
            relays: settings.relays,
            cache,
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        })
    }

    /// Returns a builder to edit settings before creating Client.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &dyn Cache {
        self.cache.as_ref()
    }

    /// Publishes a [SignedPacket] to this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    ///
    /// # Errors
    /// - Returns a [Error::NotMostRecent] if the provided signed packet is older than most recent.
    /// - Returns a [Error::RelayError] from the last responding relay, if all relays
    /// responded with non-2xx status codes.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let public_key = signed_packet.public_key();

        if let Some(current) = self.cache.get(&public_key.as_ref().into()) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(Error::NotMostRecent);
            }
        };

        self.cache.put(&public_key.as_ref().into(), signed_packet);

        self.race_publish(signed_packet).await
    }

    /// Resolve a [SignedPacket] from this client's relays.
    ///
    /// Return the first successful response, or the failure from the last responding relay.
    ///
    /// # Errors
    ///
    /// - Returns [Error::RelayError] if the relay responded with a status >= 400
    /// (except 404 in which case you should receive Ok(None)) or something wrong
    /// with the transport, transparent from [reqwest::Error].
    /// - Returns [Error::IO] if something went wrong while reading the payload.
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let cached_packet = match self.get_from_cache(public_key) {
            None => None,
            Some(signed_packet) => return Ok(Some(signed_packet)),
        };

        self.race_resolve(public_key, cached_packet).await
    }

    // === Native Race implementation ===

    #[cfg(not(target_arch = "wasm32"))]
    async fn race_publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let mut futures = JoinSet::new();

        for relay in self.relays.clone() {
            let signed_packet = signed_packet.clone();
            let this = self.clone();

            futures.spawn(async move { this.publish_to_relay(&relay, signed_packet).await });
        }

        let mut last_error = Error::EmptyListOfRelays;

        while let Some(result) = futures.join_next().await {
            match result {
                Ok(Ok(_)) => return Ok(()),
                Ok(Err(error)) => last_error = error,
                Err(joinerror) => {
                    debug!(?joinerror);
                }
            }
        }

        Err(last_error)
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn race_resolve(
        &self,
        public_key: &PublicKey,
        cached_packet: Option<SignedPacket>,
    ) -> Result<Option<SignedPacket>> {
        let mut futures = JoinSet::new();

        for relay in self.relays.clone() {
            let public_key = public_key.clone();
            let cached = cached_packet.clone();
            let this = self.clone();

            futures.spawn(async move { this.resolve_from_relay(&relay, public_key, cached).await });
        }

        let mut result: Result<Option<SignedPacket>> = Ok(None);

        while let Some(task_result) = futures.join_next().await {
            match task_result {
                Ok(Ok(Some(signed_packet))) => {
                    self.cache
                        .put(&signed_packet.public_key().as_ref().into(), &signed_packet);

                    return Ok(Some(signed_packet));
                }
                Ok(Err(error)) => result = Err(error),
                Ok(_) => {}
                Err(joinerror) => {
                    debug!(?joinerror);
                }
            }
        }

        result
    }

    // === Wasm ===

    #[cfg(target_arch = "wasm32")]
    async fn race_publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let futures = self.relays.iter().map(|relay| {
            let signed_packet = signed_packet.clone();
            let this = self.clone();

            Box::pin(async move { this.publish_to_relay(relay, signed_packet).await })
        });

        match select_ok(futures).await {
            Ok((_, _)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    #[cfg(target_arch = "wasm32")]
    async fn race_resolve(
        &self,
        public_key: &PublicKey,
        cached_packet: Option<SignedPacket>,
    ) -> Result<Option<SignedPacket>> {
        let futures = self.relays.iter().map(|relay| {
            let public_key = public_key.clone();
            let cached = cached_packet.clone();
            let this = self.clone();

            Box::pin(async move { this.resolve_from_relay(relay, public_key, cached).await })
        });

        let mut result: Result<Option<SignedPacket>> = Ok(None);

        match select_ok(futures).await {
            Ok((Some(signed_packet), _)) => {
                self.cache
                    .put(&signed_packet.public_key().as_ref().into(), &signed_packet);

                return Ok(Some(signed_packet));
            }
            Err(error) => result = Err(error),
            Ok(_) => {}
        }

        result
    }

    // === Private Methods ===

    async fn publish_to_relay(&self, relay: &str, signed_packet: SignedPacket) -> Result<Response> {
        let url = format!("{relay}/{}", signed_packet.public_key());

        self.http_client
            .put(&url)
            .body(signed_packet.to_relay_payload())
            .send()
            .await
            .map_err(|error| {
                debug!(?url, ?error, "Error response");

                Error::RelayError(error)
            })
    }

    fn get_from_cache(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        let cached_packet = self.cache.get(&public_key.as_ref().into());

        if let Some(cached) = cached_packet {
            let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");

                return Some(cached.clone());
            }

            debug!(expires_in, "Have expired signed_packet in cache.");
        } else {
            debug!("Cache miss");
        };

        None
    }

    async fn resolve_from_relay(
        &self,
        relay: &str,
        public_key: PublicKey,
        cached_packet: Option<SignedPacket>,
    ) -> Result<Option<SignedPacket>> {
        let url = format!("{relay}/{public_key}");

        match self.http_client.get(&url).send().await {
            Ok(response) => {
                if response.status() == StatusCode::NOT_FOUND {
                    debug!(?url, "SignedPacket not found");
                    return Ok(None);
                }
                if response.content_length().unwrap_or_default() > SignedPacket::MAX_BYTES {
                    debug!(?url, "Response too large");

                    return Ok(None);
                }

                let payload = response.bytes().await?;

                match SignedPacket::from_relay_payload(&public_key, &payload) {
                    Ok(signed_packet) => Ok(choose_most_recent(signed_packet, cached_packet)),
                    Err(error) => {
                        debug!(?url, ?error, "Invalid signed_packet");

                        Err(error)
                    }
                }
            }
            Err(error) => {
                debug!(?url, ?error, "Error response");

                Err(Error::RelayError(error))
            }
        }
    }
}

fn choose_most_recent(
    signed_packet: SignedPacket,
    cached_packet: Option<SignedPacket>,
) -> Option<SignedPacket> {
    if let Some(ref cached) = cached_packet {
        if signed_packet.more_recent_than(cached) {
            debug!(
                 public_key = ?signed_packet.public_key(),
                "Received more recent packet than in cache"
            );
            Some(signed_packet)
        } else {
            None
        }
    } else {
        debug!(public_key= ?signed_packet.public_key(), "Received new packet after cache miss");
        Some(signed_packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[tokio::test]
    async fn publish_resolve() {
        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let mut server = mockito::Server::new_async().await;

        let path = format!("/{}", signed_packet.public_key());

        server
            .mock("PUT", path.as_str())
            .with_header("content-type", "text/plain")
            .with_status(200)
            .create();
        server
            .mock("GET", path.as_str())
            .with_body(signed_packet.to_relay_payload())
            .create();

        let relays = vec![server.url()];
        let a = Client::builder().relays(relays.clone()).build().unwrap();
        let b = Client::builder().relays(relays).build().unwrap();

        a.publish(&signed_packet).await.unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

        assert_eq!(a.cache().len(), 1);
        assert_eq!(b.cache().len(), 1);

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[tokio::test]
    async fn not_found() {
        let keypair = Keypair::random();

        let mut server = mockito::Server::new_async().await;

        let path = format!("/{}", keypair.public_key());

        server.mock("GET", path.as_str()).with_status(404).create();

        let relays = vec![server.url()];
        let client = Client::builder().relays(relays).build().unwrap();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert!(resolved.is_none());
    }
}
