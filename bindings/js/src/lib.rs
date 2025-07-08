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

mod wasm;

pub use wasm::*; 