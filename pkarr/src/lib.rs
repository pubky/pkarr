#![doc = include_str!("../README.md")]

// TODO: add support for wasm using relays.

// Rexports
pub use bytes;
pub use simple_dns as dns;

// Modules

pub mod cache;
#[cfg(feature = "dht")]
pub mod dht;
pub mod error;
pub mod keys;
pub mod signed_packet;

// Exports
#[cfg(feature = "dht")]
pub use crate::dht::PkarrClient;
pub use crate::error::Error;
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::SignedPacket;

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;
