//! implementation of EndpointResolver trait for different clients

mod endpoint;
mod resolver;

pub use endpoint::Endpoint;
pub use resolver::EndpointsResolver;
use resolver::ResolveError;

use crate::{PublicKey, SignedPacket};

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
impl EndpointsResolver for crate::client::dht::Client {
    async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>, ResolveError> {
        self.resolve(public_key).await.map_err(|error| match error {
            crate::client::dht::ClientWasShutdown => ResolveError::ClientWasShutdown,
        })
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl EndpointsResolver for crate::client::relay::Client {
    async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>, ResolveError> {
        self.resolve(public_key)
            .await
            .map_err(ResolveError::Reqwest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dns::rdata::{A, SVCB};
    use crate::dns::{self, rdata::RData};
    use crate::{mainline::Testnet, Client, Keypair};
    use crate::{PublicKey, SignedPacket};

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
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 3, 3, Some("example.com".to_string())).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.target(), "example.com");
    }

    #[tokio::test]
    async fn empty() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let pubky = Keypair::random().public_key();

        let endpoint = client.resolve_https_endpoint(&pubky.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn max_chain_exceeded() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 4, 3, Some("example.com".to_string())).await;

        let endpoint = client.resolve_https_endpoint(&tld.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 3, 3, None).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

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
