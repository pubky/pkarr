//! implementation of Reqwest [Resolve] using Pkarr's [Endpoints](https://pkarr.org/endpoints) spec.
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
        /// [tls for pkarr domains](https://pkarr.org/tls) spec.
        fn from(client: crate::Client) -> Self {
            ::reqwest::ClientBuilder::new()
                .dns_resolver(std::sync::Arc::new(client.clone()))
                .use_preconfigured_tls(rustls::ClientConfig::from(client))
        }
    }

    #[cfg(test)]
    mod tests {
        use std::net::SocketAddr;

        use crate::{dns::rdata::SVCB, Client, Keypair, SignedPacket};

        async fn publish_server_pkarr(
            client: &Client,
            keypair: &Keypair,
            socket_addr: &SocketAddr,
        ) {
            let mut svcb = SVCB::new(0, ".".try_into().unwrap());
            svcb.set_port(socket_addr.port());

            let signed_packet = SignedPacket::builder()
                .https(".".try_into().unwrap(), svcb, 60 * 60)
                .address(".".try_into().unwrap(), socket_addr.ip(), 60 * 60)
                .sign(keypair)
                .unwrap();

            client.publish(&signed_packet, None).await.unwrap();
        }

        #[tokio::test]
        async fn reqwest_pkarr_domain() {
            use crate::mainline::Testnet;
            use std::net::TcpListener;
            use std::sync::Arc;
            use std::time::Duration;

            use axum::{routing::get, Router};
            use axum_server::tls_rustls::RustlsConfig;

            let testnet = Testnet::new_async(3).await.unwrap();

            let keypair = Keypair::random();

            {
                // Run a server on Pkarr
                let app = Router::new().route("/", get(|| async { "Hello, world!" }));
                let listener = TcpListener::bind("127.0.0.1:0").unwrap(); // Bind to any available port
                let address = listener.local_addr().unwrap();

                let client = Client::builder()
                    .bootstrap(&testnet.bootstrap)
                    .request_timeout(Duration::from_millis(100))
                    .build()
                    .unwrap();
                publish_server_pkarr(&client, &keypair, &address).await;

                println!("Server running on https://{}", keypair.public_key());

                let server = axum_server::from_tcp_rustls(
                    listener,
                    RustlsConfig::from_config(Arc::new((&keypair).into())),
                );

                tokio::spawn(server.serve(app.into_make_service()));
            }

            // Client setup
            let pkarr_client = Client::builder()
                .bootstrap(&testnet.bootstrap)
                .build()
                .unwrap();
            let reqwest = reqwest::ClientBuilder::from(pkarr_client).build().unwrap();

            // Make a request
            let response = reqwest
                .get(format!("https://{}", keypair.public_key()))
                .send()
                .await
                .unwrap();

            assert_eq!(response.status(), reqwest::StatusCode::OK);
            assert_eq!(response.text().await.unwrap(), "Hello, world!");
        }
    }
}
