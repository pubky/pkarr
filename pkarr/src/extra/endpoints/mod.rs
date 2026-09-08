//! implementation of [Endpoints](https://github.com/pubky/pkarr/blob/main/design/endpoints.md) spec.

mod endpoint;

pub use endpoint::Endpoint;

use futures_lite::{pin, Stream, StreamExt};
use genawaiter::sync::Gen;

use crate::{errors::ResolveError, PublicKey, ResolvePolicy};

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
        self.try_resolve_endpoints(qname, https)
            .filter_map(Result::ok)
    }

    /// Resolves HTTPS (`https = true`) or SVCB endpoints, preserving lookup errors.
    ///
    /// A failed alias yields an error, then resolution continues with the remaining
    /// alternatives. Consumers should keep polling if they can use another endpoint.
    pub fn try_resolve_endpoints<'a>(
        &'a self,
        qname: &'a str,
        https: bool,
    ) -> impl Stream<Item = Result<Endpoint, ResolveError>> + 'a {
        Gen::new(|co| async move {
            let mut depth = 0;
            let mut stack: Vec<Endpoint> = Vec::new();

            // Initialize the stack with endpoints from the starting domain.
            if let Ok(tld) = PublicKey::try_from(qname) {
                match self.resolve(&tld, ResolvePolicy::CacheFirst).await {
                    Ok(signed_packet) => {
                        depth += 1;
                        stack.extend(Endpoint::parse(&signed_packet, qname, https));
                    }
                    Err(error) => {
                        co.yield_(Err(error)).await;
                        return;
                    }
                }
            }

            while let Some(next) = stack.pop() {
                let current = next.target();

                // Attempt to resolve the domain as a public key.
                match PublicKey::try_from(current) {
                    Ok(tld) => match self.resolve(&tld, ResolvePolicy::CacheFirst).await {
                        Ok(signed_packet) if depth < self.max_recursion_depth => {
                            depth += 1;
                            let endpoints = Endpoint::parse(&signed_packet, current, https);

                            #[cfg(not(target_arch = "wasm32"))]
                            tracing::trace!(?qname, ?depth, ?endpoints, "resolved endpoints");
                            #[cfg(target_arch = "wasm32")]
                            log::trace!("resolved endpoints qname: {qname}, depth: {depth}, endpoints: {:?}", endpoints);

                            stack.extend(endpoints);
                        }
                        Err(error) => co.yield_(Err(error)).await,
                        _ => continue,
                    },
                    // Yield if the domain is not pointing to another Pkarr TLD domain.
                    Err(_) => co.yield_(Ok(next)).await,
                }
            }
        })
    }
}

#[derive(Debug)]
/// pkarr could not resolve endpoint
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
    use mainline::Testnet;

    use super::*;
    use crate::{Cache, ClientBuilder, InMemoryCache};
    use std::future::Future;
    use std::net::{IpAddr, Ipv4Addr};
    use std::pin::Pin;
    use std::str::FromStr;
    use std::time::Duration;
    use std::{num::NonZeroUsize, sync::Arc};

    fn cached_client_builder(packets: &[SignedPacket]) -> ClientBuilder {
        let cache = Arc::new(InMemoryCache::new(NonZeroUsize::new(8).unwrap()));
        for packet in packets {
            cache.put(&packet.public_key().into(), packet);
        }
        let mut builder = Client::builder();
        builder
            .no_dht()
            .relays(&["http://127.0.0.1:0"])
            .unwrap()
            .cache(cache);
        builder
    }

    #[tokio::test]
    async fn fallible_resolution_preserves_lookup_error() {
        let client = cached_client_builder(&[]).build().unwrap();
        let key = Keypair::random().public_key();
        let expected = client
            .resolve(&key, ResolvePolicy::CacheFirst)
            .await
            .unwrap_err();
        let name = key.to_string();
        let stream = client.try_resolve_endpoints(&name, true);
        pin!(stream);
        assert_eq!(stream.next().await.unwrap().unwrap_err(), expected);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn failed_alias_does_not_hide_alternative_endpoint() {
        let key = Keypair::random();
        let missing = Keypair::random().public_key().to_string();
        let packet = SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(1, missing.as_str().try_into().unwrap()),
                3600,
            )
            .https(
                ".".try_into().unwrap(),
                SVCB::new(2, "example.com".try_into().unwrap()),
                3600,
            )
            .sign(&key)
            .unwrap();
        let client = cached_client_builder(&[packet]).build().unwrap();
        let name = key.public_key().to_string();
        let stream = client.try_resolve_endpoints(&name, true);
        pin!(stream);
        assert!(stream.next().await.unwrap().is_err());
        assert_eq!(
            stream.next().await.unwrap().unwrap().domain(),
            Some("example.com")
        );
        assert!(stream.next().await.is_none());
        assert_eq!(
            client.resolve_https_endpoint(&name).await.unwrap().domain(),
            Some("example.com")
        );
    }

    #[tokio::test]
    async fn recursion_limit_does_not_hide_alternative_endpoint() {
        let key = Keypair::random();
        let alias = Keypair::random();
        let alias_name = alias.public_key().to_string();
        let packet = SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(1, alias_name.as_str().try_into().unwrap()),
                3600,
            )
            .https(
                ".".try_into().unwrap(),
                SVCB::new(2, "example.com".try_into().unwrap()),
                3600,
            )
            .sign(&key)
            .unwrap();
        let alias_packet = SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(1, "alias.example".try_into().unwrap()),
                3600,
            )
            .sign(&alias)
            .unwrap();
        let client = cached_client_builder(&[packet, alias_packet])
            .max_recursion_depth(1)
            .build()
            .unwrap();
        assert_eq!(
            client
                .resolve_https_endpoint(&key.public_key().to_string())
                .await
                .unwrap()
                .domain(),
            Some("example.com")
        );
    }

    #[tokio::test]
    async fn fallible_resolution_distinguishes_empty_record() {
        let key = Keypair::random();
        let packet = SignedPacket::builder().sign(&key).unwrap();
        let client = cached_client_builder(&[packet]).build().unwrap();
        let name = key.public_key().to_string();
        let stream = client.try_resolve_endpoints(&name, true);
        pin!(stream);
        assert!(stream.next().await.is_none());
    }

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
        let testnet = Testnet::builder(5).build().unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .dht(|config| {
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            })
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
        let testnet = Testnet::builder(5).build().unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .dht(|config| {
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            })
            .request_timeout(Duration::from_millis(200))
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
        let testnet = Testnet::builder(5).build().unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .dht(|config| {
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            })
            .request_timeout(Duration::from_millis(20))
            .build()
            .unwrap();

        let public_key = Keypair::random().public_key();

        let endpoint = client.resolve_https_endpoint(&public_key.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn max_recursion_exceeded() {
        let testnet = Testnet::builder(5).build().unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .dht(|config| {
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            })
            .request_timeout(Duration::from_millis(100))
            .max_recursion_depth(3)
            .build()
            .unwrap();

        let tld = generate(&client, 4, 3, Some("example.com".to_string()), vec![], None).await;

        let endpoint = client.resolve_https_endpoint(&tld.to_string()).await;

        assert!(endpoint.is_err());
    }

    #[tokio::test]
    async fn resolve_addresses() {
        let testnet = Testnet::builder(5).build().unwrap();
        let client = Client::builder()
            .no_default_network()
            .bootstrap(&testnet.bootstrap)
            .dht(|config| {
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            })
            .request_timeout(Duration::from_millis(200))
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
