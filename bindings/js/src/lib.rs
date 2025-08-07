use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

// Import DNS types for the records implementation
use ntimestamp::Timestamp;
use simple_dns::rdata::{RData, A, AAAA, HTTPS, NS, SVCB, TXT};
use simple_dns::{Name, ResourceRecord, CLASS};
use std::net::{Ipv4Addr, Ipv6Addr};

// Module declarations
mod builder;
mod client;
mod constants;
mod dns;
mod error;
mod keypair;
mod signed_packet;
mod utils;

// Re-exports
pub use builder::SignedPacketBuilder;
pub use client::Client;
pub use keypair::Keypair;
pub use signed_packet::SignedPacket;
pub use utils::Utils;
