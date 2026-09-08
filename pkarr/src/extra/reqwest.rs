//! implementation of Reqwest [Resolve] using Pkarr's [Endpoints](https://github.com/pubky/pkarr/blob/main/design/endpoints.md) and [Relays](https://github.com/pubky/pkarr/blob/main/design/relays.md) spec.
//!

use futures_lite::{pin, StreamExt};
use reqwest::dns::{Addrs, Resolve};

use crate::{Client, PublicKey};

use std::net::ToSocketAddrs;

impl Resolve for Client {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let client = self.clone();
        Box::pin(resolve(client, name))
    }
}

async fn resolve(
    client: Client,
    name: reqwest::dns::Name,
) -> Result<Addrs, Box<dyn std::error::Error + Send + Sync>> {
    let name = name.as_str();

    if PublicKey::try_from(name).is_ok() {
        let endpoints = client.try_resolve_endpoints(name, true);
        pin!(endpoints);
        let mut addrs = Vec::new();
        let mut error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

        while let Some(result) = endpoints.next().await {
            match result {
                Ok(endpoint) => match endpoint.try_to_socket_addrs() {
                    Ok(addresses) => addrs.extend(addresses),
                    Err(cause) => {
                        error.get_or_insert_with(|| Box::new(cause));
                    }
                },
                Err(cause) => {
                    error.get_or_insert_with(|| Box::new(cause));
                }
            }
        }

        if addrs.is_empty() {
            return Err(error.unwrap_or_else(|| Box::new(CouldNotResolveHost)));
        }
        tracing::trace!(?name, ?addrs, "Resolved endpoint addresses");
        Ok(Box::new(addrs.into_iter()))
    } else {
        Ok(Box::new(format!("{name}:0").to_socket_addrs()?))
    }
}

#[derive(Debug)]
/// pkarr could not resolve host.
pub struct CouldNotResolveHost;

impl std::error::Error for CouldNotResolveHost {}

impl std::fmt::Display for CouldNotResolveHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pkarr could not resolve host")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dns::rdata::SVCB, Cache, InMemoryCache, Keypair, SignedPacket};
    use std::{num::NonZeroUsize, sync::Arc};

    fn cached_client(packet: &SignedPacket) -> Client {
        let cache = Arc::new(InMemoryCache::new(NonZeroUsize::MIN));
        cache.put(&packet.public_key().into(), packet);
        Client::builder()
            .no_dht()
            .relays(&["http://127.0.0.1:0"])
            .unwrap()
            .cache(cache)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn connector_tries_alternative_endpoint() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let unused = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let unused_port = unused.local_addr().unwrap().port();
        drop(unused);
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route("/", axum::routing::get(|| async { "ok" })),
            )
            .await
            .unwrap();
        });
        let key = Keypair::random();
        let mut first = SVCB::new(1, ".".try_into().unwrap());
        first.set_port(unused_port);
        let mut second = SVCB::new(2, ".".try_into().unwrap());
        second.set_port(port);
        let packet = SignedPacket::builder()
            .https(".".try_into().unwrap(), first, 3600)
            .https(".".try_into().unwrap(), second, 3600)
            .address(".".try_into().unwrap(), "127.0.0.1".parse().unwrap(), 3600)
            .sign(&key)
            .unwrap();
        let client = cached_client(&packet);
        let name = key.public_key().to_string();
        let addresses: Vec<_> = resolve(client.clone(), name.parse().unwrap())
            .await
            .unwrap()
            .collect();
        assert_eq!(
            addresses.iter().map(|addr| addr.port()).collect::<Vec<_>>(),
            vec![unused_port, port]
        );
        let http = reqwest::Client::builder()
            .dns_resolver(Arc::new(client))
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let response = http.get(format!("http://{name}/")).send().await;
        server.abort();
        assert_eq!(response.unwrap().status(), reqwest::StatusCode::OK);
    }

    #[tokio::test]
    async fn resolver_preserves_pkarr_error() {
        let client = Client::builder()
            .no_dht()
            .relays(&["http://127.0.0.1:0"])
            .unwrap()
            .build()
            .unwrap();
        let name = Keypair::random().public_key().to_string();
        let error = resolve(client, name.parse().unwrap()).await.err().unwrap();
        assert!(error
            .downcast_ref::<crate::errors::ResolveError>()
            .is_some());
    }

    #[tokio::test]
    async fn resolver_reports_no_addresses() {
        let key = Keypair::random();
        let packet = SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(1, ".".try_into().unwrap()),
                3600,
            )
            .sign(&key)
            .unwrap();
        let client = cached_client(&packet);
        let error = resolve(client, key.public_key().to_string().parse().unwrap())
            .await
            .err()
            .unwrap();
        assert!(error.downcast_ref::<CouldNotResolveHost>().is_some());
    }
}

#[cfg(feature = "reqwest-builder")]
mod reqwest_builder {
    impl From<crate::Client> for ::reqwest::ClientBuilder {
        /// Create a [reqwest::ClientBuilder] from this Pkarr client,
        /// using it as a [dns_resolver][::reqwest::ClientBuilder::dns_resolver],
        /// and a [preconfigured_tls][::reqwest::ClientBuilder::use_preconfigured_tls] client
        /// config that uses [rustls::crypto::ring::default_provider()] and follows the
        /// [tls for pkarr domains](https://github.com/pubky/pkarr/blob/main/design/tls.md) spec.
        fn from(client: crate::Client) -> Self {
            ::reqwest::ClientBuilder::new()
                .dns_resolver(std::sync::Arc::new(client.clone()))
                .use_preconfigured_tls(rustls::ClientConfig::from(client))
        }
    }
}
