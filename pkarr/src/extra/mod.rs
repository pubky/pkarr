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
    use axum_server::{
        tls_rustls::{RustlsAcceptor, RustlsConfig},
        Server,
    };
    use mainline::Testnet;
    use rustls::{server::AlwaysResolvesServerRawPublicKeys, ServerConfig};
    use std::net::TcpListener;
    use std::sync::Arc;

    use tokio_rustls::rustls;

    use axum::{routing::get, Router};
    use std::net::SocketAddr;

    use crate::{
        dns::{rdata::SVCB, Packet},
        Client, Keypair, SignedPacket,
    };

    // #[tokio::test]
    async fn reqwest_webpki() {
        // TODO: get this working
        tracing_subscriber::fmt()
            .with_env_filter("rustls=trace")
            .init();

        // Make sure request can still make request to https://example.com
        let pkarr_client = Client::builder().build().unwrap();
        let reqwest = reqwest::ClientBuilder::from(pkarr_client).build().unwrap();

        let response = reqwest.get("https://example.com").send().await.unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }

    async fn publish_server_pkarr(client: &Client, keypair: &Keypair, socket_addr: &SocketAddr) {
        let mut packet = Packet::new_reply(1);

        let mut svcb = SVCB::new(0, ".".try_into().unwrap());

        svcb.set_port(socket_addr.port());

        packet.answers.push(crate::dns::ResourceRecord::new(
            "@".try_into().unwrap(),
            crate::dns::CLASS::IN,
            60 * 60,
            crate::dns::rdata::RData::HTTPS(svcb.into()),
        ));

        packet.answers.push(crate::dns::ResourceRecord::new(
            "@".try_into().unwrap(),
            crate::dns::CLASS::IN,
            60 * 60,
            match socket_addr.ip() {
                std::net::IpAddr::V4(ip) => crate::dns::rdata::RData::A(ip.into()),
                std::net::IpAddr::V6(ip) => crate::dns::rdata::RData::AAAA(ip.into()),
            },
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        client.publish(&signed_packet).await.unwrap();
    }

    fn server_config(keypair: &Keypair) -> ServerConfig {
        let cert_resolver =
            AlwaysResolvesServerRawPublicKeys::new(keypair.to_rpk_certified_key().into());

        ServerConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .expect("version supported by ring")
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(cert_resolver))
    }

    async fn axum_server(testnet: &Testnet, keypair: &Keypair) -> Server<RustlsAcceptor> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap(); // Bind to any available port
        let address = listener.local_addr().unwrap();

        let client = Client::builder().testnet(testnet).build().unwrap();
        publish_server_pkarr(&client, &keypair, &address).await;

        let server_config = server_config(&keypair);

        println!("Server running on https://{}", keypair.public_key());

        let server = axum_server::from_tcp_rustls(
            listener,
            RustlsConfig::from_config(Arc::new(server_config)),
        );

        server
    }

    #[tokio::test]
    async fn reqwest_pkarr_domain() {
        let testnet = Testnet::new(3).unwrap();

        let keypair = Keypair::random();

        let server = axum_server(&testnet, &keypair).await;

        // Run a server on Pkarr
        let app = Router::new().route("/", get(|| async { "Hello, world!" }));
        tokio::spawn(server.serve(app.into_make_service()));

        // Client setup
        let pkarr_client = Client::builder().testnet(&testnet).build().unwrap();
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
