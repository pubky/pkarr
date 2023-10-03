use std::net::Ipv4Addr;

use pkarr::{dns, Keypair, PkarrClient, Result, SignedPacket, Url};

#[tokio::main]
async fn main() -> Result<()> {
    let keypair = Keypair::from_secret_key(&[0; 32]);

    // Publisher
    {
        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_derp_region.iroh.").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::A(dns::rdata::A {
                address: Ipv4Addr::new(1, 1, 1, 1).into(),
            }),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;

        let client = PkarrClient::new();
        let response = client
            .relay_put(&Url::parse("http://localhost:6881").unwrap(), signed_packet)
            .await?;

        dbg!(response.status());
    }

    // Resolver
    {
        println!("\nResolving pk:{}...\n", keypair);
        let reader = PkarrClient::new();

        let signed_packet = reader
            .relay_get(
                &Url::parse("http://localhost:6882").unwrap(),
                keypair.public_key(),
            )
            .await?;

        println!("{}", signed_packet);
    }

    Ok(())
}
