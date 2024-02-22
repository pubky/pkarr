//! This example shows how to setup PkarrClient with custom dht settings.
//!
//! In this case, we are using a testnet bootstrapping nodes.

use mainline::dht::Testnet;
use pkarr::{dns, Keypair, PkarrClient, Result, SignedPacket};

fn main() -> Result<()> {
    let testnet = Testnet::new(10);

    let publickey;

    {
        let client = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();

        let keypair = Keypair::random();

        publickey = keypair.public_key();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("_foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into()?),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;

        client.publish(&signed_packet).unwrap();
    }

    dbg!(&publickey);

    let client = PkarrClient::builder().bootstrap(&testnet.bootstrap).build();

    let packet = client.resolve(publickey).unwrap();

    println!("Resolved signed packet from testnet:\n {}", packet);

    Ok(())
}
