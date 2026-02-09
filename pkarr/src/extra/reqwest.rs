//! implementation of Reqwest [Resolve] using Pkarr's [Endpoints](https://github.com/pubky/pkarr/blob/main/design/endpoints.md) and [Relays](https://github.com/pubky/pkarr/blob/main/design/relays.md) spec.
//!

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
        let endpoint = client
            .resolve_https_endpoint(name)
            .await
            .map_err(|_| CouldNotResolveHost)?;

        let addrs = endpoint.to_socket_addrs().into_iter();

        tracing::trace!(?name, ?endpoint, ?addrs, "Resolved an endpoint");

        return Ok(Box::new(addrs.into_iter()));
    };

    Ok(Box::new(format!("{name}:0").to_socket_addrs()?))
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
