//! Client implementation.

pub mod cache;

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub(crate) mod dht;
#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub use dht::{Client, Config};

#[cfg(target_arch = "wasm32")]
pub(crate) mod relay;
#[cfg(target_arch = "wasm32")]
pub use relay::{Client, Config};

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
pub mod relay;
