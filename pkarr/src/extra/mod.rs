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
    fn from(client: crate::Client) -> Self {
        ::reqwest::ClientBuilder::new()
            .dns_resolver(std::sync::Arc::new(client.clone()))
            .use_preconfigured_tls(rustls::ClientConfig::from(client))
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "reqwest-builder"))]
impl From<crate::client::relay::Client> for ::reqwest::ClientBuilder {
    fn from(client: crate::client::relay::Client) -> Self {
        ::reqwest::ClientBuilder::new()
            .dns_resolver(std::sync::Arc::new(client.clone()))
            .use_preconfigured_tls(rustls::ClientConfig::from(client))
    }
}

#[cfg(test)]
mod tests {
    use rustls::{server::AlwaysResolvesServerRawPublicKeys, ServerConfig};
    use std::sync::Arc;

    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio_rustls::{rustls, TlsAcceptor};

    use crate::{Client, Keypair};

    #[tokio::test]
    async fn reqwest_webpki() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");

        // Make sure request can still make request to https://example.com
        let pkarr_client = Client::builder().build().unwrap();
        let reqwest = reqwest::ClientBuilder::from(pkarr_client).build().unwrap();

        let response = reqwest.get("https://example.com").send().await.unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }

    // #[tokio::test]
    async fn reqwest_pkarr_domain() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .expect("Failed to install rustls crypto provider");

        // Server setup
        let server_keypair = Keypair::random();
        let cert_resolver =
            AlwaysResolvesServerRawPublicKeys::new(server_keypair.to_rpk_certified_key().into());

        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(cert_resolver));
        let acceptor = TlsAcceptor::from(Arc::new(server_config));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap(); // Bind to any available port
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let acceptor = acceptor.clone();

                tokio::spawn(async move {
                    let mut stream = acceptor.accept(stream).await.unwrap();

                    stream
                        .write_all(
                            &b"HTTP/1.0 200 ok\r\n\
                        Connection: close\r\n\
                        Content-length: 12\r\n\
                        \r\n\
                        Hello world!"[..],
                        )
                        .await
                        .unwrap();

                    stream.shutdown().await.unwrap();
                });
            }
        });

        // Client setup
        let pkarr_client = Client::builder().build().unwrap();
        let reqwest = reqwest::ClientBuilder::from(pkarr_client).build().unwrap();

        // Make a request
        let response = reqwest
            .get(format!("https://{}", addr))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), "Hello world!");
    }
}
