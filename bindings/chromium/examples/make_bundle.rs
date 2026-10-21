//! Write a complete service response for the standalone C++ example.

use std::time::{Duration, SystemTime};

use base64::{Engine, engine::general_purpose::STANDARD};
use pkarr::{Keypair, SignedPacket, dns::Name, dns::rdata::SVCB};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected output file path")?;
    let key = Keypair::random();
    let root = Name::new_unchecked(".");
    let mut packet = SignedPacket::builder()
        .https(root.clone(), SVCB::new(1, root.clone()), 60)
        .address(root, "203.0.113.10".parse()?, 60)
        .sign(&key)?;
    // The server has cached this exact packet for ten seconds. Serving it must
    // not refresh that observation time at the client.
    packet.set_last_seen(&(SystemTime::now() - Duration::from_secs(10)).into());
    let bundle = serde_json::json!({
        "schema": 1,
        "packets": [STANDARD.encode(packet.serialize())],
    });
    std::fs::write(path, serde_json::to_vec(&bundle)?)?;
    println!("{}", key.to_z32());
    Ok(())
}
