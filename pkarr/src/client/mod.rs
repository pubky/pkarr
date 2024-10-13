//! Client implementation.

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub mod dht;
#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
pub use dht::Client;

#[cfg(target_arch = "wasm32")]
pub mod relay;
#[cfg(target_arch = "wasm32")]
pub use relay::Client;

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
pub mod relay;
