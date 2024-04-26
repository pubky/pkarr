#![doc = include_str!("../README.md")]

// TODO: add support for wasm using relays.
// TODO: allow custom Cache with traits.

// Rexports
pub use bytes;
pub use simple_dns as dns;

// Modules

#[cfg(feature = "async")]
pub mod async_client;
#[cfg(feature = "dht")]
pub mod cache;
#[cfg(feature = "dht")]
pub mod client;
pub mod error;
pub mod keys;
pub mod signed_packet;

// Exports
#[cfg(feature = "dht")]
pub use crate::client::PkarrClient;
pub use crate::error::Error;
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::SignedPacket;

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;
