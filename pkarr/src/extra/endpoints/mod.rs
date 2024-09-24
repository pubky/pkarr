mod endpoint;

use crate::{
    error::{Error, Result},
    Client, PublicKey,
};

use endpoint::Endpoint;

const DEFAULT_MAX_CHAIN_LENGTH: u8 = 3;

#[derive(Debug, Clone)]
pub struct EndpointResolver {
    pkarr: Client,
    max_chain_length: u8,
}

impl EndpointResolver {
    pub fn new(pkarr: Client, max_chain_length: u8) -> Self {
        EndpointResolver {
            pkarr,
            max_chain_length,
        }
    }

    /// Resolve a `qname` to an alternative [Endpoint] as defined in [RFC9460](https://www.rfc-editor.org/rfc/rfc9460#name-terminology).
    ///
    /// A `qname` is can be either a regular domain name for HTTPS endpoints,
    /// or it could use Attrleaf naming pattern for cusotm protcol. For example:
    /// `_foo.example.com` for `foo://example.com`.
    async fn resolve_endpoint(&self, qname: &str) -> Result<Endpoint> {
        let target = qname;
        // TODO: cache the result of this function?

        let is_svcb = target.starts_with('_');

        let mut step = 0;
        let mut svcb: Option<Endpoint> = None;

        loop {
            let current = svcb.clone().map_or(target.to_string(), |s| s.target);
            if let Ok(tld) = PublicKey::try_from(current.clone()) {
                if let Ok(Some(signed_packet)) = self.pkarr.resolve(&tld).await {
                    if step >= self.max_chain_length {
                        break;
                    };
                    step += 1;

                    // Choose most prior SVCB record
                    svcb = Endpoint::find(&signed_packet, &current, is_svcb);

                    // TODO: support wildcard?
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if let Some(svcb) = svcb {
            if PublicKey::try_from(svcb.target.as_str()).is_err() {
                return Ok(svcb);
            }
        }

        Err(Error::Generic(format!(
            "Failed to find an endopint {}",
            target
        )))
    }
}

impl From<&Client> for EndpointResolver {
    fn from(pkarr: &Client) -> Self {
        pkarr.clone().into()
    }
}

impl From<Client> for EndpointResolver {
    /// Creates [EndpointResolver] from [Client] and default settings
    fn from(pkarr: Client) -> Self {
        Self::new(pkarr, DEFAULT_MAX_CHAIN_LENGTH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dns::rdata::{A, SVCB};
    use crate::dns::{self, rdata::RData};
    use crate::SignedPacket;
    use crate::{mainline::Testnet, Keypair};

    use std::future::Future;
    use std::pin::Pin;

    fn generate_subtree(
        client: Client,
        depth: u8,
        branching: u8,
        domain: Option<String>,
    ) -> Pin<Box<dyn Future<Output = PublicKey>>> {
        Box::pin(async move {
            let keypair = Keypair::random();

            let mut packet = dns::Packet::new_reply(0);

            for _ in 0..branching {
                let mut svcb = SVCB::new(0, ".".try_into().unwrap());

                if depth == 0 {
                    svcb.priority = 1;
                    svcb.set_port((branching) as u16 * 1000);

                    if let Some(target) = &domain {
                        let target: &'static str = Box::leak(target.clone().into_boxed_str());
                        svcb.target = target.try_into().unwrap()
                    }
                } else {
                    let target =
                        generate_subtree(client.clone(), depth - 1, branching, domain.clone())
                            .await
                            .to_string();
                    let target: &'static str = Box::leak(target.into_boxed_str());
                    svcb.target = target.try_into().unwrap();
                };

                packet.answers.push(dns::ResourceRecord::new(
                    dns::Name::new("@").unwrap(),
                    dns::CLASS::IN,
                    3600,
                    RData::HTTPS(svcb.into()),
                ));
            }

            if depth == 0 {
                packet.answers.push(dns::ResourceRecord::new(
                    dns::Name::new("@").unwrap(),
                    dns::CLASS::IN,
                    3600,
                    RData::A(A { address: 10 }),
                ));
            }

            let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();
            client.publish(&signed_packet).await.unwrap();

            keypair.public_key()
        })
    }

    fn generate(
        client: Client,
        depth: u8,
        branching: u8,
        domain: Option<String>,
    ) -> Pin<Box<dyn Future<Output = PublicKey>>> {
        generate_subtree(client, depth - 1, branching, domain)
    }

    #[tokio::test]
    async fn resolve_endpoints() {
        let testnet = Testnet::new(3);
        let pkarr = Client::builder().testnet(&testnet).build().unwrap();

        let resolver: EndpointResolver = (&pkarr).into();
        let tld = generate(pkarr, 3, 3, Some("example.com".to_string())).await;

        let endpoint = resolver.resolve_endpoint(&tld.to_string()).await.unwrap();
        assert_eq!(endpoint.target, "example.com");
    }

    #[tokio::test]
    async fn max_chain_exceeded() {
        let testnet = Testnet::new(3);
        let pkarr = Client::builder().testnet(&testnet).build().unwrap();

        let resolver: EndpointResolver = (&pkarr).into();

        let tld = generate(pkarr, 4, 3, Some("example.com".to_string())).await;

        let endpoint = resolver.resolve_endpoint(&tld.to_string()).await;

        assert!(endpoint.is_err());
        // TODO: test error correctly

        // assert_eq!(
        //     match endpoint {
        //         Err(error) => error.to_string(),
        //         _ => "".to_string(),
        //     },
        //     Error::Generic(tld.to_string())
        // )
    }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::new(3);
        let pkarr = Client::builder().testnet(&testnet).build().unwrap();

        let resolver: EndpointResolver = (&pkarr).into();
        let tld = generate(pkarr, 3, 3, None).await;

        let endpoint = resolver.resolve_endpoint(&tld.to_string()).await.unwrap();
        assert_eq!(endpoint.target, ".");
        assert_eq!(endpoint.port, 3000);
        assert_eq!(
            endpoint
                .to_socket_addrs()
                .unwrap()
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
            vec!["0.0.0.10:3000"]
        );
        dbg!(&endpoint);
    }
}
