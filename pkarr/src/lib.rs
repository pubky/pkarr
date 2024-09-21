#![doc = include_str!("../README.md")]
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

macro_rules! if_dht {
    ($($item:item)*) => {$(
        #[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
        $item
    )*}
}

// Modules
mod error;
mod keys;
mod signed_packet;

// Common exports
pub use crate::error::{Error, Result};
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::{system_time, SignedPacket};

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

// Rexports
pub use bytes;
pub use simple_dns as dns;

if_dht! {
    mod cache;
    mod dht;

    pub use dht::{ClientBuilder, Client, Settings};
    pub use cache::{PkarrCache, PkarrCacheKey, InMemoryPkarrCache};

    // Rexports
    pub use mainline;
}

#[cfg(feature = "relay")]
pub mod relay;
