//! Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).

use futures::future::select_ok;
use reqwest::Client;
use url::Url;

use crate::{Error, PublicKey, Result, SignedPacket};

pub const DEFAULT_RELAYS: [&str; 1] = ["https://relay.pkarr.org"];

#[derive(Debug, Clone)]
/// Pkarr client for publishing and resolving [SignedPacket]s over [relays](https://pkarr.org/relays).
pub struct PkarrRelayClient {
    http_client: Client,
    relays: Vec<Url>,
}

impl Default for PkarrRelayClient {
    fn default() -> Self {
        Self::new(DEFAULT_RELAYS.map(|s| s.try_into().unwrap()).to_vec()).unwrap()
    }
}

impl PkarrRelayClient {
    pub fn new(relays: Vec<Url>) -> Result<Self> {
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
            http_client: Client::new(),
            relays,
        })
    }

    /// Publishes a [SignedPacket] to this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
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
                    dbg!("publish error");
                    Err(Error::RelayErrorResponse(
                        url.to_string(),
                        response.status(),
                        response.text().await.unwrap_or_default(),
                    ))
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
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
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
    use super::PkarrRelayClient;
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

        let a = PkarrRelayClient::new(relays.clone()).unwrap();
        let b = PkarrRelayClient::new(relays).unwrap();

        let _ = a.publish(&signed_packet).await;

        let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }
}
