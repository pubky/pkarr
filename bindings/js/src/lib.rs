//! JavaScript/WebAssembly bindings for Pkarr
//!
//! This crate provides JavaScript bindings for the Pkarr library,
//! allowing JavaScript and Node.js applications to publish and resolve
//! DNS records over the Mainline DHT through Pkarr relays.
//!
//! # Usage
//!
//! ```javascript
//! const { Client, Keypair, SignedPacket } = require('pkarr-js');
//!
//! // Create a new keypair and client
//! const keypair = new Keypair();
//! const client = new Client();
//!
//! // Build a DNS packet
//! const builder = SignedPacket.builder();
//! builder.addTxtRecord("_service", "pkarr=v1.0", 3600);
//! builder.addARecord("www", "192.168.1.1", 3600);
//!
//! // Sign and publish
//! const packet = builder.buildAndSign(keypair);
//! await client.publish(packet);
//!
//! // Resolve later
//! const resolved = await client.resolve(keypair.public_key_string());
//! console.log('Records:', resolved.records);
//! ```

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

// Import core library types with aliases to avoid name conflicts with our WASM wrapper structs
// Without aliases, Rust would be confused about which "Keypair" or "SignedPacket" we're referring to
// This ensures our WASM wrappers properly delegate to the native types instead of creating circular references
use pkarr::{
    Keypair as NativeKeypair, PublicKey, SignedPacket as NativeSignedPacket,
    SignedPacketBuilder as NativeSignedPacketBuilder, RelaysClient,
};

// Import DNS types for the records implementation
use ntimestamp::Timestamp;
use simple_dns::rdata::{RData, A, AAAA, HTTPS, NS, SVCB, TXT};
use simple_dns::{Name, ResourceRecord, CLASS};
use std::net::{Ipv4Addr, Ipv6Addr};

// Module declarations
mod builder;
mod client;
mod constants;
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
