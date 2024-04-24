#![doc = include_str!("../README.md")]

// TODO: Deploy relay / resolver and add it as a default resolver
// TODO: examine errors (failed to publish, failed to bind socket, unused errors...)
// TODO: logs (info for binding, debug for steps)
// TODO: HTTP relay should return some caching headers.
// TODO: add server settings to mainline DhtSettings
// TODO: better documentation especially resolvers.
// TODO: add support for wasm using relays.
// TODO: allow custom Cache with traits.

// Rexports
pub use bytes;
pub use simple_dns as dns;

// Modules

#[cfg(feature = "async")]
pub mod async_client;
#[cfg(feature = "dht")]
mod cache;
#[cfg(feature = "dht")]
mod client;
mod error;
mod keys;
mod signed_packet;

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
