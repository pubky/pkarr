//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use std::{
    io::Read,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    thread,
};

use flume::Receiver;
use lru::LruCache;
use tracing::debug;
use ureq::Agent;

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
    /// Custom [ureq::Agent]
    pub http_client: Agent,
}

impl Default for RelaySettings {
    fn default() -> Self {
        Self {
            relays: DEFAULT_RELAYS.map(|s| s.into()).to_vec(),
            cache_size: NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            http_client: ureq::Agent::new(),
        }
    }
}

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct PkarrRelayClient {
    http_client: Agent,
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
    pub fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let mut last_error = Error::EmptyListOfRelays;

        while let Ok(response) = self.publish_inner(signed_packet)?.recv() {
            match response {
                Ok(_) => return Ok(()),
                Err(error) => {
                    last_error = error;
                }
            }
        }

        Err(last_error)
    }

    /// Resolve a [SignedPacket] from this client's relays.
    ///
    /// Return the first successful response, or the failure from the last responding relay.
    ///
    /// # Errors
    ///
    /// - Returns [Error::RelayError] if the relay responded with a status >= 400
    /// (except 404 in which case you should receive Ok(None)) or something wrong
    /// with the transport, transparent from [ureq::Error].
    /// - Returns [Error::IO] if something went wrong while reading the payload.
    pub fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        if let Some(signed_packet) = self.resolve_inner(public_key).recv()?? {
            self.cache
                .lock()
                .unwrap()
                .put(public_key.clone(), signed_packet.clone());

            return Ok(Some(signed_packet));
        };

        Ok(None)
    }

    // === Private Methods ===

    pub(crate) fn publish_inner(
        &self,
        signed_packet: &SignedPacket,
    ) -> Result<Receiver<Result<()>>> {
        let public_key = signed_packet.public_key();
        let mut cache = self.cache.lock().unwrap();

        if let Some(current) = cache.get(&public_key) {
            if current.timestamp() > signed_packet.timestamp() {
                return Err(Error::NotMostRecent);
            }
        };

        cache.put(public_key.to_owned(), signed_packet.clone());
        drop(cache);

        let (sender, receiver) = flume::bounded::<Result<()>>(1);

        for relay in self.relays.clone() {
            let url = format!("{relay}/{public_key}");
            let http_client = self.http_client.clone();
            let sender = sender.clone();
            let signed_packet = signed_packet.clone();

            thread::spawn(move || {
                match http_client
                    .put(&url)
                    .send_bytes(&signed_packet.to_relay_payload())
                {
                    Ok(_) => {
                        let _ = sender.send(Ok(()));
                    }
                    Err(error) => {
                        debug!(?url, ?error, "Error response");
                        let _ = sender.send(Err(Error::RelayError(Box::new(error))));
                    }
                }
            });
        }

        Ok(receiver)
    }

    pub(crate) fn resolve_inner(
        &self,
        public_key: &PublicKey,
    ) -> Receiver<Result<Option<SignedPacket>>> {
        let mut cache = self.cache.lock().unwrap();

        let cached_packet = cache.get(public_key);

        let (sender, receiver) = flume::bounded::<Result<Option<SignedPacket>>>(1);

        if let Some(cached) = cached_packet {
            let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);

            if expires_in > 0 {
                debug!(expires_in, "Have fresh signed_packet in cache.");
                let _ = sender.send(Ok(Some(cached.clone())));

                return receiver;
            }

            debug!(expires_in, "Have expired signed_packet in cache.");
        } else {
            debug!("Cache mess");
        };

        for relay in self.relays.clone() {
            let url = format!("{relay}/{public_key}");
            let http_client = self.http_client.clone();
            let public_key = public_key.clone();
            let cached_packet = cached_packet.cloned();
            let sender = sender.clone();

            thread::spawn(move || match http_client.get(&url).call() {
                Ok(response) => {
                    let mut reader = response.into_reader();
                    let mut payload = vec![];

                    if let Err(err) = reader.read_to_end(&mut payload) {
                        let _ = sender.send(Err(err.into()));
                    } else {
                        match SignedPacket::from_relay_payload(&public_key, &payload.into()) {
                            Ok(signed_packet) => {
                                let new_packet = if let Some(ref cached) = cached_packet {
                                    if signed_packet.more_recent_than(cached) {
                                        debug!(
                                            ?public_key,
                                            "Received more recent packet than in cache"
                                        );
                                        Some(signed_packet)
                                    } else {
                                        None
                                    }
                                } else {
                                    debug!(?public_key, "Received new packet after cache miss");
                                    Some(signed_packet)
                                };

                                let _ = sender.send(Ok(new_packet));
                            }
                            Err(error) => {
                                debug!(?url, ?error, "Invalid signed_packet");
                                let _ = sender.send(Err(error));
                            }
                        };
                    }
                }
                Err(ureq::Error::Status(404, _)) => {
                    dbg!(404);
                    debug!(?url, "SignedPacket not found");
                    let _ = sender.send(Ok(None));
                }
                Err(error) => {
                    debug!(?url, ?error, "Error response");
                    let _ = sender.send(Err(Error::RelayError(Box::new(error))));
                }
            });
        }

        receiver
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dns, Keypair, SignedPacket};

    #[test]
    fn publish_resolve() {
        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let mut server = mockito::Server::new();

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

        a.publish(&signed_packet).unwrap();

        let resolved = b.resolve(&keypair.public_key()).unwrap().unwrap();

        assert_eq!(a.cache().lock().unwrap().len(), 1);
        assert_eq!(b.cache().lock().unwrap().len(), 1);

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }

    #[test]
    fn not_found() {
        let keypair = Keypair::random();

        let mut server = mockito::Server::new();

        let path = format!("/{}", keypair.public_key());

        server.mock("GET", path.as_str()).with_status(404).create();

        let relays: Vec<String> = vec![server.url()];
        let settings = RelaySettings {
            relays,
            ..RelaySettings::default()
        };

        let client = PkarrRelayClient::new(settings.clone()).unwrap();

        let resolved = client.resolve(&keypair.public_key()).unwrap();

        assert!(resolved.is_none());
    }
}
