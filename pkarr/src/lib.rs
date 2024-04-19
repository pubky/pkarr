#![doc = include_str!("../README.md")]
#![allow(unused)]

// TODO add support for wasm using relays.

// Rexports
pub use bytes;
pub use simple_dns as dns;

// Modules

mod client;
mod error;
mod keys;
mod signed_packet;

// Exports
pub use crate::client::PkarrClient;
pub use crate::error::Error;
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::SignedPacket;

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;
