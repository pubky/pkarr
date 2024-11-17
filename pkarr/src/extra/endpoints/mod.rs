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
    use simple_dns::rdata::AAAA;

    use super::*;
    use crate::dns::rdata::{A, SVCB};
    use crate::dns::{self, rdata::RData};
    use crate::{mainline::Testnet, Client, Keypair};
    use crate::{PublicKey, SignedPacket};

    use std::future::Future;
    use std::net::IpAddr;
    use std::pin::Pin;
    use std::str::FromStr;

    // TODO: test SVCB too.

    fn generate_subtree(
        client: Client,
        depth: u8,
        branching: u8,
        domain: Option<String>,
        ips: Vec<IpAddr>,
        port: Option<u16>,
    ) -> Pin<Box<dyn Future<Output = PublicKey>>> {
        Box::pin(async move {
            let keypair = Keypair::random();

            let mut packet = dns::Packet::new_reply(0);

            for _ in 0..branching {
                let mut svcb = SVCB::new(0, ".".try_into().unwrap());

                if depth == 0 {
                    svcb.priority = 1;

                    if let Some(port) = port {
                        svcb.set_port(port);
                    }

                    if let Some(target) = &domain {
                        let target: &'static str = Box::leak(target.clone().into_boxed_str());
                        svcb.target = target.try_into().unwrap()
                    }

                    for ip in ips.clone() {
                        packet.answers.push(dns::ResourceRecord::new(
                            dns::Name::new("@").unwrap(),
                            dns::CLASS::IN,
                            3600,
                            match ip {
                                IpAddr::V4(address) => RData::A(A {
                                    address: address.into(),
                                }),
                                IpAddr::V6(address) => RData::AAAA(AAAA {
                                    address: address.into(),
                                }),
                            },
                        ));
                    }
                } else {
                    let target = generate_subtree(
                        client.clone(),
                        depth - 1,
                        branching,
                        domain.clone(),
                        ips.clone(),
                        port,
                    )
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

            let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();
            client.publish(&signed_packet).await.unwrap();

            keypair.public_key()
        })
    }

    /// depth of (3): A -> B -> C
    /// branch of (2): A -> B0,  A ->  B1
    /// domain, ips, and port are all at the end (C, or B1)
    fn generate(
        client: &Client,
        depth: u8,
        branching: u8,
        domain: Option<String>,
        ips: Vec<IpAddr>,
        port: Option<u16>,
    ) -> Pin<Box<dyn Future<Output = PublicKey>>> {
        generate_subtree(client.clone(), depth - 1, branching, domain, ips, port)
    }

    #[tokio::test]
    async fn direct_endpoint_resolution() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 1, 1, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.domain(), "example.com");
        assert_eq!(endpoint.public_key(), &tld);
    }

    #[tokio::test]
    async fn resolve_endpoints() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(&client, 3, 3, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.domain(), "example.com");
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

        let tld = generate(&client, 4, 3, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client.resolve_https_endpoint(&tld.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder().testnet(&testnet).build().unwrap();

        let tld = generate(
            &client,
            3,
            3,
            None,
            vec![IpAddr::from_str("0.0.0.10").unwrap()],
            Some(3000),
        )
        .await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.domain(), ".");
        assert_eq!(
            endpoint
                .to_socket_addrs()
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
            vec!["0.0.0.10:3000"]
        );
    }
}
