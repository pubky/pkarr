//! Client implementation.

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
mod dht;
#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub use dht::{Client, ClientBuilder, Settings};

#[cfg(target_arch = "wasm32")]
mod relay;
#[cfg(target_arch = "wasm32")]
pub use relay::{Client, ClientBuilder, Settings};

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
pub mod relay;
