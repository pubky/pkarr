//! Run an HTTP server listening on a Pkarr domain
//!
//! This server will _not_ be accessible from other networks
//! unless the provided IP is public and the port number is forwarded.

use tracing::Level;
use tracing_subscriber;

use axum::{response::Html, routing::get, Router};
use axum_server::bind;
use std::net::{SocketAddr, ToSocketAddrs};

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

    let app = Router::new().route("/", get(handler));

    let addr = format!("{}:{}", cli.ip, cli.port)
        .to_socket_addrs()?
        .next()
        .ok_or(anyhow::anyhow!(
            "Could not convert IP and port to socket addresses"
        ))?;

    publish_server_pkarr(&addr).await;

    bind(addr).serve(app.into_make_service()).await?;

    Ok(())
}

// Simple handler that responds with "Hello, World!"
async fn handler() -> Html<&'static str> {
    Html("Hello, World!")
}

async fn publish_server_pkarr(socket_addr: &SocketAddr) {
    let client = Client::builder().build().unwrap();

    let keypair = Keypair::random();

    let mut packet = Packet::new_reply(1);

    let mut svcb = SVCB::new(0, ".".try_into().unwrap());

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

    println!("Server running on https://{}", keypair.public_key());

    client.publish(&signed_packet).await.unwrap();
}
