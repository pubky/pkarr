use bytes::Bytes;
use pkarr::{PublicKey, SignedPacket};
use std::path::Path;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <public_key> <file_path>", args[0]);
        eprintln!(
            "Example: {} 57ipp7cyxoghrxqk1koh3n5bk5ke1hsz9oo3cn93n1r3htxcpz1o packet.bin",
            args[0]
        );
        std::process::exit(1);
    }

    // Parse public key from first argument
    let public_key_str = &args[1];
    let pk = PublicKey::try_from(public_key_str.as_str())
        .map_err(|e| format!("Invalid public key '{}': {}", public_key_str, e))?;

    // Get file path from second argument
    let file_path = Path::new(&args[2]);

    if !file_path.exists() {
        eprintln!(
            "File {} does not exist",
            file_path.to_str().unwrap_or("None")
        );
        std::process::exit(1);
    }

    println!("Public key: {}", pk);
    println!("Reading file: {}", file_path.to_str().unwrap_or("None"));

    let file = fs::read(file_path)?;
    let packet = Bytes::from(file);

    match deserialize_from_relay_payload(&pk, &packet) {
        Ok(packet) => {
            println!("\nSuccessfully verified and deserialized packet!");
            print_packet_info(&packet);
        }
        Err(e) => {
            eprintln!("Failed to deserialize packet: {}", e);
        }
    }

    Ok(())
}

fn deserialize_from_relay_payload(
    public_key: &PublicKey,
    payload: &Bytes,
) -> Result<SignedPacket, Box<dyn std::error::Error>> {
    let packet = SignedPacket::from_relay_payload(public_key, payload)?;
    Ok(packet)
}

fn print_packet_info(packet: &SignedPacket) {
    println!("\nPublic Key: {}", packet.public_key());
    println!(
        "Timestamp: {} ({})",
        packet.timestamp().as_u64(),
        packet.timestamp().format_http_date()
    );
    println!("Last Seen: {} seconds ago", packet.elapsed());
    println!("Signature: {}", packet.signature());

    println!("\nDNS Records:");
    for record in packet.all_resource_records() {
        println!(
            "  {} {} IN {:?} {}",
            record.name,
            record.ttl,
            record.class,
            format_rdata(&record.rdata)
        );
    }

    let min_ttl = 300; // 5 minutes
    let max_ttl = 86400; // 24 hours

    println!(
        "\nCheck if TTL in between {}s min, {}s max",
        min_ttl, max_ttl
    );
    if packet.is_expired(min_ttl, max_ttl) {
        println!("\nWarning: This packet is expired!");
    } else {
        println!(
            "\nPacket expires in {} seconds",
            packet.expires_in(min_ttl, max_ttl)
        );
    }
}

fn format_rdata(rdata: &pkarr::dns::rdata::RData) -> String {
    use pkarr::dns::rdata::{RData, A, AAAA};
    use std::net::{Ipv4Addr, Ipv6Addr};

    match rdata {
        RData::A(A { address }) => format!("A {}", Ipv4Addr::from(*address)),
        RData::AAAA(AAAA { address }) => format!("AAAA {}", Ipv6Addr::from(*address)),
        RData::CNAME(name) => format!("CNAME {:?}", name),
        RData::TXT(txt) => {
            let text: Result<String, _> = txt.clone().try_into();
            format!(
                "TXT \"{}\"",
                text.unwrap_or_else(|_| "__INVALID_TXT_VALUE__".to_string())
            )
        }
        _ => format!("{:?}", rdata),
    }
}
