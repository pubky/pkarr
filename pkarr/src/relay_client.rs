//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use lru::LruCache;
use reqwest::{Client, Response, StatusCode};
use tokio::task::JoinSet;
use tracing::debug;

use crate::{
    Error, PublicKey, Result, SignedPacket, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL,
    DEFAULT_MINIMUM_TTL, DEFAULT_RELAYS,
};

#[derive(Debug, Clone)]
/// [PkarrRelayClient]'s settings
pub struct RelaySettings {
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
    pub http_client: Client,
}

impl Default for RelaySettings {
    fn default() -> Self {
        Self {
            relays: DEFAULT_RELAYS.map(|s| s.into()).to_vec(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            http_client: Client::new(),
        }
    }
}

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct PkarrRelayClient {
    http_client: Client,
    relays: Vec<String>,
    cache: Arc<Mutex<LruCache<PublicKey, SignedPacket>>>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl Default for PkarrRelayClient {
    fn default() -> Self {
        Self::new(RelaySettings::default()).unwrap()
    }
}

impl PkarrRelayClient {
    pub fn new(settings: RelaySettings) -> Result<Self> {
        if settings.relays.is_empty() {
            return Err(Error::EmptyListOfRelays);
        }

        Ok(Self {
            http_client: settings.http_client,
            relays: settings.relays,
            cache: Arc::new(Mutex::new(LruCache::new(settings.cache_size))),
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        })
    }

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &Mutex<LruCache<PublicKey, SignedPacket>> {
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

        // Let the compiler know we are dropping the cache before await
        {
            let mut cache = self.cache.lock().unwrap();

            if let Some(current) = cache.get(&public_key) {
                if current.timestamp() > signed_packet.timestamp() {
                    return Err(Error::NotMostRecent);
                }
            };

            cache.put(public_key.to_owned(), signed_packet.clone());
        }

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

    async fn race_publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let mut futures = JoinSet::new();

        for relay in self.relays.clone() {
            let signed_packet = signed_packet.clone();
            let this = self.clone();

            futures.spawn(async move { this.publish_to_relay(relay, signed_packet).await });
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

            futures.spawn(async move { this.resolve_from_relay(relay, public_key, cached).await });
        }

        let mut result: Result<Option<SignedPacket>> = Ok(None);

        while let Some(task_result) = futures.join_next().await {
            match task_result {
                Ok(Ok(Some(signed_packet))) => {
                    let mut cache = self.cache.lock().unwrap();

                    cache.put(signed_packet.public_key(), signed_packet.clone());

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

    // === Private Methods ===

    async fn publish_to_relay(
        &self,
        relay: String,
        signed_packet: SignedPacket,
    ) -> Result<Response> {
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
        let mut cache = self.cache.lock().unwrap();

        let cached_packet = cache.get(public_key);

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
        relay: String,
        public_key: PublicKey,
        cached_packet: Option<SignedPacket>,
    ) -> Result<Option<SignedPacket>> {
        let url = format!("{relay}/{public_key}");

        match self.http_client.get(&url).send().await {
            Ok(mut response) => {
                if response.status() == StatusCode::NOT_FOUND {
                    debug!(?url, "SignedPacket not found");
                    return Ok(None);
                }

                let payload = read(&mut response).await;

                match SignedPacket::from_relay_payload(&public_key, &payload.into()) {
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

async fn read(response: &mut Response) -> Vec<u8> {
    let mut total_size = 0;
    let mut payload = Vec::new();

    while let Ok(Some(chunk)) = response.chunk().await {
        total_size += chunk.len();
        payload.extend_from_slice(&chunk);
        if total_size >= SignedPacket::MAX_BYTES {
            break;
        }
    }
    payload
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

        let relays: Vec<String> = vec![server.url()];
        let settings = RelaySettings {
            relays,
            ..RelaySettings::default()
        };

        let a = PkarrRelayClient::new(settings.clone()).unwrap();
        let b = PkarrRelayClient::new(settings).unwrap();

        a.publish(&signed_packet).await.unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

        assert_eq!(a.cache().lock().unwrap().len(), 1);
        assert_eq!(b.cache().lock().unwrap().len(), 1);

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[tokio::test]
    async fn not_found() {
        let keypair = Keypair::random();

        let mut server = mockito::Server::new_async().await;

        let path = format!("/{}", keypair.public_key());

        server.mock("GET", path.as_str()).with_status(404).create();

        let relays: Vec<String> = vec![server.url()];
        let settings = RelaySettings {
            relays,
            ..RelaySettings::default()
        };

        let client = PkarrRelayClient::new(settings.clone()).unwrap();

        let resolved = client.resolve(&keypair.public_key()).await.unwrap();

        assert!(resolved.is_none());
    }
}
