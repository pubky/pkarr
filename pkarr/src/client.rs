//! Client implementation.

pub mod cache;

mod builder;
#[cfg(any(feature = "relays", target_arch = "wasm32"))]
mod relays;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Client, ClientBuilder};

#[cfg(target_arch = "wasm32")]
pub(crate) mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{Client, ClientBuilder, Config};
