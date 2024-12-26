#[cfg(feature = "endpoints")]
pub mod endpoints;

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-resolve"))]
pub mod reqwest;

#[cfg(all(not(target_arch = "wasm32"), feature = "tls"))]
pub mod tls;

#[cfg(all(not(target_arch = "wasm32"), feature = "lmdb-cache"))]
pub mod lmdb_cache;

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-builder"))]
impl From<crate::Client> for ::reqwest::ClientBuilder {
    /// Create a [reqwest::ClientBuilder][::reqwest::ClientBuilder] from this Pkarr client,
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

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-builder"))]
impl From<crate::client::relay::Client> for ::reqwest::ClientBuilder {
    /// Create a [reqwest::ClientBuilder][::reqwest::ClientBuilder] from this Pkarr client,
    /// using it as a [dns_resolver][::reqwest::ClientBuilder::dns_resolver],
    /// and a [preconfigured_tls][::reqwest::ClientBuilder::use_preconfigured_tls] client
    /// config that uses [rustls::crypto::ring::default_provider()] and follows the
    /// [tls for pkarr domains](https://pkarr.org/tls) spec.
    fn from(client: crate::client::relay::Client) -> Self {
        ::reqwest::ClientBuilder::new()
            .dns_resolver(std::sync::Arc::new(client.clone()))
            .use_preconfigured_tls(rustls::ClientConfig::from(client))
    }
}

#[cfg(test)]
mod tests {
    use mainline::Testnet;
    use std::net::SocketAddr;
    use std::net::TcpListener;
    use std::sync::Arc;

    use axum::{routing::get, Router};
    use axum_server::tls_rustls::RustlsConfig;

    use crate::{dns::rdata::SVCB, Client, Keypair, SignedPacket};

    async fn publish_server_pkarr(client: &Client, keypair: &Keypair, socket_addr: &SocketAddr) {
        let mut svcb = SVCB::new(0, ".".try_into().unwrap());
        svcb.set_port(socket_addr.port());

        let signed_packet = SignedPacket::builder()
            .https(".".try_into().unwrap(), svcb, 60 * 60)
            .address(".".try_into().unwrap(), socket_addr.ip(), 60 * 60)
            .sign(&keypair)
            .unwrap();

        client.publish(&signed_packet).await.unwrap();
    }

    #[tokio::test]
    async fn reqwest_pkarr_domain() {
        let testnet = Testnet::new(3).unwrap();

        let keypair = Keypair::random();

        {
            // Run a server on Pkarr
            let app = Router::new().route("/", get(|| async { "Hello, world!" }));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap(); // Bind to any available port
            let address = listener.local_addr().unwrap();

            let client = Client::builder()
                .bootstrap(&testnet.bootstrap)
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
