//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use std::{num::NonZeroUsize, sync::Arc};

use futures::{future::select_ok, lock::Mutex};
use lru::LruCache;
use reqwest::Client;
use tracing::{debug, instrument};
use url::Url;

use crate::{
    Error, PublicKey, Result, SignedPacket, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL,
    DEFAULT_MINIMUM_TTL,
};

pub const DEFAULT_RELAYS: [&str; 1] = ["https://relay.pkarr.org"];

#[derive(Debug, Clone)]
/// [PkarrRelayClient]'s settings
pub struct RelaySettings {
    pub relays: Option<Vec<Url>>,
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
    pub http_client: Option<reqwest::Client>,
}

impl Default for RelaySettings {
    fn default() -> Self {
        Self {
            relays: Some(DEFAULT_RELAYS.map(|s| s.try_into().unwrap()).to_vec()),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            http_client: Some(Client::default()),
        }
    }
}

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct PkarrRelayClient {
    http_client: Client,
    relays: Vec<Url>,
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
        let relays = settings.relays.unwrap_or_default();

        if relays.is_empty() {
            return Err(Error::EmptyListOfRelays);
        }

        for url in &relays {
            let scheme = url.scheme();
            if url.cannot_be_a_base() || (scheme != "http" && scheme != "https") {
                return Err(Error::InvalidRelayUrl(url.clone()));
            }
        }

        Ok(Self {
            http_client: settings.http_client.unwrap_or_default(),
            relays,
            cache: Arc::new(Mutex::new(LruCache::new(settings.cache_size))),
            minimum_ttl: settings.minimum_ttl,
            maximum_ttl: settings.maximum_ttl,
        })
    }

    /// Publishes a [SignedPacket] to this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    ///
    /// # Errors
    /// - Returns a [Error::NotMostRecent] if the provided signed packet is older than most recent.
    /// - Returns a [Error::RelayErrorResponse] from the last responding relay, if all relays
    /// responded with non-2xx status codes.
    #[instrument(skip(self))]
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let public_key = signed_packet.public_key();
        let mut cache = self.cache.lock().await;

        if let Some(current) = cache.get(&public_key) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(Error::NotMostRecent);
            }
        };

        cache.put(public_key, signed_packet.clone());
        drop(cache);

        let futures = self.relays.iter().cloned().map(|relay| {
            Box::pin(async {
                let mut url = relay;
                url.path_segments_mut()
                    .unwrap()
                    .push(&signed_packet.public_key().to_z32());

                let response = self
                    .http_client
                    .put(url.to_owned())
                    .body(signed_packet.to_relay_payload())
                    .send()
                    .await?;

                if response.status().is_success() {
                    Ok(())
                } else {
                    let error = Err(Error::RelayErrorResponse(
                        url.to_string(),
                        response.status(),
                        response.text().await.unwrap_or_default(),
                    ));
                    debug!(?error);
                    error
                }
            })
        });

        match select_ok(futures).await {
            Ok((response, _)) => Ok(response),
            Err(e) => Err(e),
        }
    }

    /// Resolve a [SignedPacket] from this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    #[instrument(skip(self))]
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let mut cache = self.cache.lock().await;

        let cached_packet = cache.get(public_key);

        if let Some(cached) = cached_packet {
            let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");
                return Ok(Some(cached.clone()));
            }

            debug!(expires_in, "Have expired signed_packet in cache.");
        } else {
            debug!("Cache mess");
        }

        let futures = self
            .relays
            .iter()
            .cloned()
            .map(|relay| {
                Box::pin(async {
                    let mut url = relay;
                    url.path_segments_mut().unwrap().push(&public_key.to_z32());

                    let response = self.http_client.get(url.to_owned()).send().await?;

                    if response.status().is_success() {
                        let bytes = response.bytes().await?;
                        Ok(Some(SignedPacket::from_relay_payload(public_key, &bytes)?))
                    } else {
                        Err(Error::RelayErrorResponse(
                            url.to_string(),
                            response.status(),
                            response.text().await.unwrap_or_default(),
                        ))
                    }
                })
            })
            .collect::<Vec<_>>();

        match select_ok(futures).await {
            Ok((response, _)) => Ok(response),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dns, Keypair, SignedPacket};
    use url::Url;

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

        let relays: Vec<Url> = vec![server.url().as_str().try_into().unwrap()];
        let settings = RelaySettings {
            relays: Some(relays),
            ..RelaySettings::default()
        };

        let a = PkarrRelayClient::new(settings.clone()).unwrap();
        let b = PkarrRelayClient::new(settings).unwrap();

        let _ = a.publish(&signed_packet).await;

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }
}
