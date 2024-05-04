#![doc = include_str!("../README.md")]

macro_rules! if_dht {
    ($($item:item)*) => {$(
        #[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
        $item
    )*}
}

// Rexports
pub use bytes;
pub use simple_dns as dns;

// Modules
mod error;
mod keys;
#[cfg(feature = "relay")]
mod relay_client;
mod signed_packet;

// Common exports
pub use crate::error::{Error, Result};
pub use crate::keys::{Keypair, PublicKey};
pub use crate::signed_packet::{system_time, SignedPacket};

if_dht! {
    mod cache;
    mod client;
    #[cfg(feature = "async")]
    mod client_async;

    pub use client::{PkarrClientBuilder, PkarrClient, Settings};
    #[cfg(feature = "async")]
    pub use client_async::PkarrClientAsync;

    pub use cache::{PkarrCache, PkarrCacheKey, InMemoryPkarrCache};

    pub use mainline;

    /// Default minimum TTL: 5 minutes
    pub const DEFAULT_MINIMUM_TTL: u32 = 300;
    /// Default maximum TTL: 24 hours
    pub const DEFAULT_MAXIMUM_TTL: u32 = 24 * 60 * 60;
    /// Default cache size: 1000
    pub const DEFAULT_CACHE_SIZE: usize = 1000;
    /// Default resolvers
    pub const DEFAULT_RESOLVERS: [&str; 1] = ["resolver.pkarr.org:6881"];
}

#[cfg(feature = "relay")]
pub use relay_client::{PkarrRelayClient, DEFAULT_RELAYS};
