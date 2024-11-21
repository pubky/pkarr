//! Run an HTTP server listening on a Pkarr domain
//!
//! This server will _not_ be accessible from other networks
//! unless the provided IP is public and the port number is forwarded.

use tracing::Level;
use tracing_subscriber;

use axum::{routing::get, Router};
use axum_server::tls_rustls::RustlsConfig;

use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;

use clap::Parser;

use pkarr::{
    dns::{rdata::SVCB, Packet},
    Client, Keypair, SignedPacket,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// IP address to listen on
    ip: String,
    /// Port number to listen no
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    let cli = Cli::parse();

    let addr = format!("{}:{}", cli.ip, cli.port)
        .to_socket_addrs()?
        .next()
        .ok_or(anyhow::anyhow!(
            "Could not convert IP and port to socket addresses"
        ))?;

    let keypair = Keypair::random();

    let client = Client::builder().build()?;

    // Run a server on Pkarr
    println!("Server listening on {addr}");

    // You should republish this every time the socket address change
    // and once an hour otherwise.
    publish_server_pkarr(&client, &keypair, &addr).await;

    println!("Server running on https://{}", keypair.public_key());

    let server = axum_server::bind_rustls(
        addr,
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
