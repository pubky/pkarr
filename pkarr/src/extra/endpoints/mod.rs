//! EndpointResolver trait for different clients

// mod async_iter;
mod endpoint;

use crate::{
    error::{Error, Result},
    Client, PublicKey, SignedPacket,
};

pub use endpoint::Endpoint;

const DEFAULT_MAX_CHAIN_LENGTH: u8 = 3;

pub trait EndpointResolver {
    fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> impl std::future::Future<Output = Result<Option<SignedPacket>>>;

    fn resolve_endpoint(&self, qname: &str) -> impl std::future::Future<Output = Result<Endpoint>>
    where
        Self: std::marker::Sync,
    {
        async move {
            let target = qname;
            // TODO: cache the result of this function?

            let is_svcb = target.starts_with('_');

            let mut step = 0;
            let mut svcb: Option<Endpoint> = None;

            loop {
                let current = svcb
                    .clone()
                    .map_or(target.to_string(), |s| s.target().to_string());
                if let Ok(tld) = PublicKey::try_from(current.clone()) {
                    if let Ok(Some(signed_packet)) = self.resolve(&tld).await {
                        if step >= DEFAULT_MAX_CHAIN_LENGTH {
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
                if PublicKey::try_from(svcb.target()).is_err() {
                    return Ok(svcb);
                }
            }

            Err(Error::Generic(format!(
                "Failed to find an endopint {}",
                target
            )))
        }
    }
}

impl EndpointResolver for Client {
    async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        self.resolve(public_key).await
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl EndpointResolver for crate::relay::Client {
    async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        self.resolve(public_key).await
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
        client: &Client,
        depth: u8,
        branching: u8,
        domain: Option<String>,
    ) -> Pin<Box<dyn Future<Output = PublicKey>>> {
        generate_subtree(client.clone(), depth - 1, branching, domain)
    }

    #[tokio::test]
    async fn resolve_endpoints() {
        let testnet = Testnet::new(3);
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 3, 3, Some("example.com".to_string())).await;

        let endpoint = client.resolve_endpoint(&tld.to_string()).await.unwrap();
        assert_eq!(endpoint.target(), "example.com");
    }

    // TODO: Test max_chain_exceeded
    // #[tokio::test]
    // async fn max_chain_exceeded() {
    //     let testnet = Testnet::new(3);
    //     let pkarr = Client::builder().testnet(&testnet).build().unwrap();
    //
    //     let resolver: EndpointResolver = (&pkarr).into();
    //
    //     let tld = generate(pkarr, 4, 3, Some("example.com".to_string())).await;
    //
    //     let endpoint = resolver.resolve_endpoint(&tld.to_string()).await;
    //
    //     assert!(endpoint.is_err());
    //     // TODO: test error correctly
    //
    //     // assert_eq!(
    //     //     match endpoint {
    //     //         Err(error) => error.to_string(),
    //     //         _ => "".to_string(),
    //     //     },
    //     //     Error::Generic(tld.to_string())
    //     // )
    // }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::new(3);
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 3, 3, None).await;

        let endpoint = client.resolve_endpoint(&tld.to_string()).await.unwrap();
        assert_eq!(endpoint.target(), ".");
        assert_eq!(endpoint.port(), Some(3000));
        assert_eq!(
            endpoint
                .to_socket_addrs()
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
            vec!["0.0.0.10:3000"]
        );
        dbg!(&endpoint);
    }
}
