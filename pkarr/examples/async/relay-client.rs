
//! This example shows how to publish and resolve [ResourceRecord]s to and from a Pkarr
//! [relay](https://github.com/Nuhvi/pkarr/blob/main/design/relays.md).
//!
//! In this example, we publish both an A and CNAME records as well as a TXT record representing
//! [Iroh](https://iroh.computer/) DERP regions used by Peer with the same Ed25519 key.
//!
//! For the purpose of this example we use the default Pkarr relay for both
//! publishing and resolving, but you can override `relay1` and `relay2` to
//! with your own locally running relays.
//!
//! Publishing takes a few seconds as the relay only responds after sending the PUT
//! request to all the nearest nodes to the Pkarr key.
//!
//! Resolving on the other hand is optimistic and the relay will retun the first
//! response it gets.
//!
//! run this example from the project root:
//!     $ cargo run --example relay-client

use std::time::Instant;

use pkarr::{dns, url::Url, Keypair, PkarrClient, Result, SignedPacket, DEFAULT_PKARR_RELAY};

#[tokio::main]
async fn main() -> Result<()> {
    let keypair = Keypair::random();

    let relay1 = DEFAULT_PKARR_RELAY;
    let relay2 = DEFAULT_PKARR_RELAY;

    // Publisher
    {
        println!("\nPublishing pk:{}...\n", keypair);
        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_derp_region.iroh.").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::A(dns::rdata::A {
                address: std::net::Ipv4Addr::new(52, 30, 229, 248).into(),
            }),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_derp_region.iroh.").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::CNAME(
                dns::Name::new("eu.derp.iroh.network")
                    .expect("is valid name")
                    .into(),
            ),
        ));
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_derp_region.iroh.").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("url=eu.derp.iroh.network:33000".try_into()?),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;

        let client = PkarrClient::new();
        client.relay_put(&Url::parse(relay1).unwrap(), signed_packet).await?;

        println!("Published pk:{}", keypair);
    }

    // Resolver
    {
        println!("\nResolving pk:{}...\n", keypair);
        let reader = PkarrClient::new();

        let instant = Instant::now();

        let signed_packet = reader.relay_get(&Url::parse(relay2).unwrap(), keypair.public_key()).await?;

        println!("Resolved in {:?} \n{}", instant.elapsed(), signed_packet);
    }

    Ok(())
}
