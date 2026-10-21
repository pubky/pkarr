//! Resolve a locally signed packet synchronously, without network I/O or an executor.
//!
//! For async packet sources, implement `AsyncPacketProvider` and call
//! `derive_resolution_async(&request, &provider).await` instead.

use std::{collections::HashMap, sync::Arc};

use pkarr::{dns::rdata::SVCB, dns::Name, Keypair, SignedPacket};
use pkresolve::{derive_resolution, HttpProtocol, ResolutionRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keypair = Keypair::random();
    let root_label = Name::new(".")?;
    let packet = SignedPacket::builder()
        .https(root_label.clone(), SVCB::new(1, root_label.clone()), 60)
        .address(root_label, "203.0.113.10".parse()?, 60)
        .sign(&keypair)?;
    // A packet map implements the synchronous PacketProvider trait directly.
    let provider = HashMap::from([(keypair.public_key(), Arc::new(packet))]);
    let request =
        ResolutionRequest::new(keypair.to_z32().parse()?, 443, vec![HttpProtocol::Http11])?;
    let result = derive_resolution(&request, &provider)?;
    for candidate in &result.candidates {
        for address in &candidate.addresses {
            println!(
                "{}; SNI {}; required raw TLS key {}",
                std::net::SocketAddr::new(*address, candidate.port.get()),
                result.service,
                result.service.public_key(),
            );
        }
    }
    Ok(())
}
