//! This example shows how to publish [ResourceRecord]s directly to the DHT.
//!
//! Change the `Keypair::random()` to your own keypair to publish your own records.
//! Change the `packet.answers` to your own records.
//!
//! run this example from the project root:
//!     $ cargo run --example publish

use std::time::Instant;

use pkarr::{dns, Keypair, PkarrClient, Result, SignedPacket};

fn main() -> Result<()> {
    let client = PkarrClient::new();

    let keypair = Keypair::random();

    let mut packet = dns::Packet::new_reply(0);
    packet.answers.push(dns::ResourceRecord::new(
        dns::Name::new("_foo").unwrap(),
        dns::CLASS::IN,
        30,
        dns::rdata::RData::TXT("bar".try_into()?),
    ));

    let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;

    let instant = Instant::now();

    println!("\nPublishing {} ...", keypair.to_uri_string());

    match client.publish(&signed_packet) {
        Ok(metadata) => {
            println!(
                "\nSuccessfully published in {:?} to {} nodes",
                instant.elapsed(),
                metadata.stored_at().len(),
            );
        }
        Err(err) => {
            println!("\nFailed to publish {} \n {}", keypair.public_key(), err);
        }
    };

    Ok(())
}
