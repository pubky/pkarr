//! Client implementation.

pub mod cache;

#[cfg(any(feature = "relays", target_arch = "wasm32"))]
mod shared;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Client, ClientBuilder, Config};

#[cfg(target_arch = "wasm32")]
pub(crate) mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{Client, ClientBuilder, Config};
