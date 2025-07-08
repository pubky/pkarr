//! WASM bindings for pkarr relay functions

use console_error_panic_hook;
use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

// Import core library types with aliases to avoid name conflicts with our WASM wrapper structs
// Without aliases, Rust would be confused about which "Keypair" or "SignedPacket" we're referring to
// This ensures our WASM wrappers properly delegate to the native types instead of creating circular references
use pkarr::{
    Keypair as NativeKeypair, PublicKey, SignedPacket as NativeSignedPacket,
    SignedPacketBuilder as NativeSignedPacketBuilder,
};

#[cfg(feature = "relays")]
use pkarr::RelaysClient;

// Import DNS types for the records implementation
use ntimestamp::Timestamp;
use simple_dns::rdata::{RData, A, AAAA, HTTPS, NS, SVCB, TXT};
use simple_dns::{Name, ResourceRecord, CLASS};
use std::net::{Ipv4Addr, Ipv6Addr};

#[cfg(feature = "relays")]
use futures_lite;

// Additional imports needed by submodules
#[cfg(feature = "relays")]
use url;

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
