#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

// Modules
mod base;
mod client;
mod error;

// Exports
pub use base::cache::{Cache, CacheKey, InMemoryCache};
pub use base::keys::{Keypair, PublicKey};
pub use base::signed_packet::SignedPacket;
pub use base::time::system_time;
pub use error::{Error, Result};

/// Default minimum TTL: 5 minutes
pub const DEFAULT_MINIMUM_TTL: u32 = 300;
/// Default maximum TTL: 24 hours
pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
/// Default cache size: 1000
pub const DEFAULT_CACHE_SIZE: usize = 1000;
/// Default [relay](https://pkarr.org/relays)s
pub const DEFAULT_RELAYS: [&str; 2] = ["https://relay.pkarr.org", "https://pkarr.pubky.app"];
/// Default [resolver](https://pkarr.org/resolvers)s
pub const DEFAULT_RESOLVERS: [&str; 2] = ["resolver.pkarr.org:6881", "pkarr.pubky.app:6881"];

pub use client::{Client, ClientBuilder, Settings};

// Rexports
pub use bytes;
pub use simple_dns as dns;

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub use mainline;
