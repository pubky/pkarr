//! implementation of Reqwest [Resolve] using Pkarr's [Endpoints](https://github.com/pubky/pkarr/blob/main/design/endpoints.md) and [Relays](https://github.com/pubky/pkarr/blob/main/design/relays.md) spec.
//!

use futures_lite::{pin, StreamExt};
use reqwest::dns::{Addrs, Resolve};

use crate::{errors::ResolveError, extra::endpoints::Endpoint, Client, PublicKey};

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;
use tokio::time::{timeout_at, Instant};

const BACKUP_RESOLUTION_TIMEOUT: Duration = Duration::from_millis(100);

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
        let endpoints = client
            .try_resolve_endpoints(name, true)
            .then(resolve_endpoint);
        pin!(endpoints);
        let mut addrs = Vec::new();
        let mut error: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        let mut backup_deadline = None;

        loop {
            let next = match backup_deadline {
                // Reqwest cannot connect until this future returns its address list.
                Some(deadline) => timeout_at(deadline, endpoints.next()).await.ok().flatten(),
                None => endpoints.next().await,
            };
            let Some(result) = next else { break };
            match result {
                Ok(addresses) => {
                    addrs.extend(addresses);
                    if !addrs.is_empty() && backup_deadline.is_none() {
                        backup_deadline = Some(Instant::now() + BACKUP_RESOLUTION_TIMEOUT);
                    }
                }
                Err(cause) => {
                    error.get_or_insert(cause);
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

async fn resolve_endpoint(
    result: Result<Endpoint, ResolveError>,
) -> Result<Vec<SocketAddr>, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = result?;
    if let Some(domain) = endpoint.domain() {
        return Ok(
            tokio::net::lookup_host((domain, endpoint.port().unwrap_or_default()))
                .await?
                .collect(),
        );
    }
    Ok(endpoint.try_to_socket_addrs()?)
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

    #[rstest::rstest]
    #[case::direct(".")]
    #[case::domain("localhost")]
    #[tokio::test]
    async fn connector_tries_alternative_endpoint(#[case] target: &str) {
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
        let mut second = SVCB::new(2, target.try_into().unwrap());
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
        assert!(addresses.len() >= 2);
        assert_eq!(addresses[0].port(), unused_port);
        assert!(addresses[1..].iter().all(|address| address.port() == port));
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
    async fn healthy_primary_does_not_wait_for_backup_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let primary = tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route("/", axum::routing::get(|| async { "ok" })),
            )
            .await
            .unwrap();
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay_url = format!("http://{}", listener.local_addr().unwrap());
        let relay = tokio::spawn(async move {
            axum::serve(
                listener,
                axum::Router::new().route(
                    "/{key}",
                    axum::routing::get(std::future::pending::<axum::http::StatusCode>),
                ),
            )
            .await
            .unwrap();
        });
        let key = Keypair::random();
        let alias = Keypair::random().public_key().to_string();
        let mut first = SVCB::new(1, ".".try_into().unwrap());
        first.set_port(address.port());
        let packet = SignedPacket::builder()
            .https(".".try_into().unwrap(), first, 3600)
            .https(
                ".".try_into().unwrap(),
                SVCB::new(2, alias.as_str().try_into().unwrap()),
                3600,
            )
            .address(".".try_into().unwrap(), address.ip(), 3600)
            .sign(&key)
            .unwrap();
        let cache = Arc::new(InMemoryCache::new(NonZeroUsize::MIN));
        cache.put(&key.public_key().into(), &packet);
        let client = Client::builder()
            .no_dht()
            .relays(&[relay_url])
            .unwrap()
            .cache(cache)
            .reqwest_client(reqwest::Client::builder().no_proxy().build().unwrap())
            .request_timeout(std::time::Duration::from_secs(4))
            .build()
            .unwrap();
        let http = reqwest::Client::builder()
            .no_proxy()
            .dns_resolver(Arc::new(client))
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .unwrap();
        let response = http
            .get(format!("http://{}/", key.public_key()))
            .send()
            .await;
        relay.abort();
        primary.abort();
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
