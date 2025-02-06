//! implementation of EndpointResolver trait for different clients

mod endpoint;

pub use endpoint::Endpoint;

use futures_lite::{pin, Stream, StreamExt};
use genawaiter::sync::Gen;

use crate::PublicKey;

impl crate::Client {
    /// Returns an async stream of [HTTPS][crate::dns::rdata::RData::HTTPS] [Endpoint]s
    pub fn resolve_https_endpoints<'a>(
        &'a self,
        qname: &'a str,
    ) -> impl Stream<Item = Endpoint> + 'a {
        self.resolve_endpoints(qname, true)
    }

    /// Returns an async stream of [SVCB][crate::dns::rdata::RData::SVCB] [Endpoint]s
    pub fn resolve_svcb_endpoints<'a>(
        &'a self,
        qname: &'a str,
    ) -> impl Stream<Item = Endpoint> + 'a {
        self.resolve_endpoints(qname, false)
    }

    /// Helper method that returns the first [HTTPS][crate::dns::rdata::RData::HTTPS] [Endpoint] in the Async stream from [Self::resolve_https_endpoints]
    pub async fn resolve_https_endpoint(
        &self,
        qname: &str,
    ) -> Result<Endpoint, CouldNotResolveEndpoint> {
        let stream = self.resolve_https_endpoints(qname);

        pin!(stream);

        match stream.next().await {
            Some(endpoint) => Ok(endpoint),
            None => {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::trace!(?qname, "failed to resolve endpoint");
                #[cfg(target_arch = "wasm32")]
                log::trace!("failed to resolve endpoint {qname}");

                Err(CouldNotResolveEndpoint)
            }
        }
    }

    /// Helper method that returns the first [SVCB][crate::dns::rdata::RData::SVCB] [Endpoint] in the Async stream from [Self::resolve_svcb_endpoints]
    pub async fn resolve_svcb_endpoint(
        &self,
        qname: &str,
    ) -> Result<Endpoint, CouldNotResolveEndpoint> {
        let stream = self.resolve_https_endpoints(qname);

        pin!(stream);

        match stream.next().await {
            Some(endpoint) => Ok(endpoint),
            None => Err(CouldNotResolveEndpoint),
        }
    }

    /// Returns an async stream of either [HTTPS][crate::dns::rdata::RData::HTTPS] or [SVCB][crate::dns::rdata::RData::SVCB] [Endpoint]s
    pub fn resolve_endpoints<'a>(
        &'a self,
        qname: &'a str,
        https: bool,
    ) -> impl Stream<Item = Endpoint> + 'a {
        Gen::new(|co| async move {
            let mut depth = 0;
            let mut stack: Vec<Endpoint> = Vec::new();

            // Initialize the stack with endpoints from the starting domain.
            if let Ok(tld) = PublicKey::try_from(qname) {
                if let Some(signed_packet) = self.resolve(&tld).await {
                    depth += 1;
                    stack.extend(Endpoint::parse(&signed_packet, qname, https));
                }
            }

            while let Some(next) = stack.pop() {
                let current = next.target();

                // Attempt to resolve the domain as a public key.
                match PublicKey::try_from(current) {
                    Ok(tld) => match self.resolve(&tld).await {
                        Some(signed_packet) if depth < self.0.max_recursion_depth => {
                            depth += 1;
                            let endpoints = Endpoint::parse(&signed_packet, current, https);

                            #[cfg(not(target_arch = "wasm32"))]
                            tracing::trace!(?qname, ?depth, ?endpoints, "resolved endpoints");
                            #[cfg(target_arch = "wasm32")]
                            log::trace!("resolved endpoints qname: {qname}, depth: {depth}, endpoints: {:?}", endpoints);

                            stack.extend(endpoints);
                        }
                        _ => break, // Stop on resolution failure or recursion depth exceeded.
                    },
                    // Yield if the domain is not pointing to another Pkarr TLD domain.
                    Err(_) => co.yield_(next).await,
                }
            }
        })
    }
}

#[derive(Debug)]
pub struct CouldNotResolveEndpoint;

impl std::error::Error for CouldNotResolveEndpoint {}

impl std::fmt::Display for CouldNotResolveEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pkarr could not resolve endpoint")
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {

    use crate::dns::rdata::SVCB;
    use crate::{Client, Keypair};
    use crate::{PublicKey, SignedPacket};

    use std::future::Future;
    use std::net::IpAddr;
    use std::pin::Pin;
    use std::str::FromStr;
    use std::time::Duration;

    use mainline::Testnet;

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

            let mut builder = SignedPacket::builder();

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
                        builder = builder.address(".".try_into().unwrap(), ip, 3600);
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

                builder = builder.https(".".try_into().unwrap(), svcb, 3600);
            }

            let signed_packet = builder.sign(&keypair).unwrap();

            client.publish(&signed_packet, None).await.unwrap();

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
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let tld = generate(&client, 1, 1, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.domain(), Some("example.com"));
        assert_eq!(endpoint.public_key(), &tld);
    }

    #[tokio::test]
    async fn resolve_endpoints() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

        let tld = generate(&client, 3, 3, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client
            .resolve_https_endpoint(&tld.to_string())
            .await
            .unwrap();

        assert_eq!(endpoint.domain(), Some("example.com"));
    }

    #[tokio::test]
    async fn empty() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .request_timeout(Duration::from_millis(20))
            .build()
            .unwrap();

        let pubky = Keypair::random().public_key();

        let endpoint = client.resolve_https_endpoint(&pubky.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn max_recursion_exceeded() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .max_recursion_depth(3)
            .build()
            .unwrap();

        let tld = generate(&client, 4, 3, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client.resolve_https_endpoint(&tld.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::new(3).unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .build()
            .unwrap();

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

        assert_eq!(endpoint.target(), ".");
        assert_eq!(endpoint.domain(), None);
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
