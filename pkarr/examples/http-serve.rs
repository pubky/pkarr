//! Run an HTTP server listening on a Pkarr domain
//!
//! This server will _not_ be accessible from other networks
//! unless the provided IP is public and the port number is forwarded.

use tracing::Level;
use tracing_subscriber;

use axum::{routing::get, Router};
use axum_server::tls_rustls::RustlsConfig;

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use pkarr::{
    dns::{rdata::SVCB, Packet},
    Client, Keypair, SignedPacket,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let keypair = Keypair::random();

    // Run a server on Pkarr
    let listener = TcpListener::bind("127.0.0.1:0").unwrap(); // Bind to any available port
    let address = listener.local_addr()?;
    println!("Server listening on {address}");

    let client = Client::builder().build()?;
    // You should republish this every time the socket address change
    // and once an hour otherwise.
    publish_server_pkarr(&client, &keypair, &address).await;

    println!("Server running on https://{}", keypair.public_key());

    let server = axum_server::from_tcp_rustls(
        listener,
        RustlsConfig::from_config(Arc::new(keypair.to_rpk_rustls_server_config())),
    );

    let app = Router::new().route("/", get(|| async { "Hello, world!" }));
    server.serve(app.into_make_service()).await?;

    Ok(())
}

async fn publish_server_pkarr(client: &Client, keypair: &Keypair, socket_addr: &SocketAddr) {
    let mut packet = Packet::new_reply(1);

    let mut svcb = SVCB::new(0, ".".try_into().expect("infallible"));

    svcb.set_port(socket_addr.port());

    packet.answers.push(pkarr::dns::ResourceRecord::new(
        "@".try_into().unwrap(),
        pkarr::dns::CLASS::IN,
        60 * 60,
        pkarr::dns::rdata::RData::HTTPS(svcb.into()),
    ));

    packet.answers.push(pkarr::dns::ResourceRecord::new(
        "@".try_into().unwrap(),
        pkarr::dns::CLASS::IN,
        60 * 60,
        match socket_addr.ip() {
            std::net::IpAddr::V4(ip) => pkarr::dns::rdata::RData::A(ip.into()),
            std::net::IpAddr::V6(ip) => pkarr::dns::rdata::RData::AAAA(ip.into()),
        },
    ));

    let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

    client.publish(&signed_packet).await.unwrap();
}
