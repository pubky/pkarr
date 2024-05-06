//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use std::sync::Mutex;

use lru::LruCache;

use crate::{Error, PkarrRelayClient, PublicKey, Result, SignedPacket};

pub struct PkarrRelayClientAsync(PkarrRelayClient);

impl PkarrRelayClient {
    pub fn as_async(self) -> PkarrRelayClientAsync {
        PkarrRelayClientAsync(self)
    }
}

impl PkarrRelayClientAsync {
    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> &Mutex<LruCache<PublicKey, SignedPacket>> {
        self.0.cache()
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
        let mut last_error = Error::EmptyListOfRelays;

        while let Ok(response) = self.0.publish_inner(signed_packet)?.recv_async().await {
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
    /// - Returns [Error::RelayError] if the relay responded with a status >= 400 or something
    /// wring with the transport, transparent from [ureq::Error].
    /// - Returns [Error::IO] if something went wrong while reading the payload.
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let mut last_result = Ok(None);

        while let Ok(response) = self.0.resolve_inner(public_key).recv_async().await {
            match response {
                Ok(Some(signed_packet)) => {
                    self.cache()
                        .lock()
                        .unwrap()
                        .put(public_key.clone(), signed_packet.clone());

                    return Ok(Some(signed_packet));
                }
                result => last_result = result,
            }
        }

        last_result
    }
}

#[cfg(test)]
mod tests {
    use crate::{dns, Keypair, PkarrRelayClient, RelaySettings, SignedPacket};

    #[test]
    fn publish_resolve() {
        async fn test() {
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

            let a = PkarrRelayClient::new(settings.clone()).unwrap().as_async();
            let b = PkarrRelayClient::new(settings).unwrap().as_async();

            a.publish(&signed_packet).await.unwrap();

            let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

            assert_eq!(a.cache().lock().unwrap().len(), 1);
            assert_eq!(b.cache().lock().unwrap().len(), 1);

            assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
        }

        futures::executor::block_on(test());
    }
}
