//! This example shows how to publish [ResourceRecord]s directly to the DHT.
//!
//! Change the `Keypair::random()` to your own keypair to publish your own records.
//! Change the `packet.answers` to your own records.
//!
//! run this example from the project root:
//!     $ cargo run --example publish

use hickory_proto::op::Message;
use hickory_proto::rr::{rdata, DNSClass, Name, RData, Record, RecordType};
use tracing::Level;
use tracing_subscriber;

use std::time::Instant;

use pkarr::{Keypair, PkarrClient, Result, SignedPacket};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    let client = PkarrClient::builder().build().unwrap();

    let keypair = Keypair::random();

    let mut packet = Message::new();
    let mut record = Record::with(Name::from_ascii("_foo").unwrap(), RecordType::TXT, 30);
    record.set_dns_class(DNSClass::IN);
    record.set_data(Some(RData::TXT(rdata::TXT::new(vec!["bar".to_string()]))));
    packet.add_answer(record);

    let signed_packet = SignedPacket::from_packet(&keypair, &packet)?;

    let instant = Instant::now();

    println!("\nPublishing {} ...", keypair.public_key());

    match client.publish(&signed_packet) {
        Ok(()) => {
            println!(
                "\nSuccessfully published {} in {:?}",
                keypair.public_key(),
                instant.elapsed(),
            );
        }
        Err(err) => {
            println!("\nFailed to publish {} \n {}", keypair.public_key(), err);
        }
    };

    Ok(())
}
