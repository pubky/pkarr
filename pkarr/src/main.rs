use std::net::Ipv4Addr;

use pkarr::{dns, Keypair, RelayClient, Result, SignedPacket};

#[tokio::main]
async fn main() -> Result<()> {
    let keypair = Keypair::from_secret_key(&[0; 32]);

    // let mut packet = dns::Packet::new_reply(0);
    // packet.answers.push(dns::ResourceRecord::new(
    //     dns::Name::new("_derp_region.iroh.").unwrap(),
    //     dns::CLASS::IN,
    //     30,
    //     dns::rdata::RData::A(dns::rdata::A {
    //         address: Ipv4Addr::new(1, 1, 1, 1).into(),
    //     }),
    // ));
    //
    // let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;
    //
    // let alice = RelayClient::new("http://localhost:6881")?;
    // alice.put(signed_packet).await?;

    println!("\nResolving pk:{}...\n", keypair);
    let bob = RelayClient::new("http://localhost:6882")?;

    let public_key = keypair.public_key();

    let signed_packet = bob.get(public_key).await?;
    println!("{}", signed_packet);

    Ok(())
}
