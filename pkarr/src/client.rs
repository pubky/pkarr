//! Client implementation.

pub mod cache;

#[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
mod dht;
#[cfg(not(target_arch = "wasm32"))]
mod relays;
mod shared;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Client, Config};

#[cfg(target_arch = "wasm32")]
pub(crate) mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{Client, Config};
