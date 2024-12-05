//! Client implementation.

pub mod cache;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{Client, Settings};

// TODO: add wasm again!
// #[cfg(target_arch = "wasm32")]
// pub(crate) mod wasm;
// #[cfg(target_arch = "wasm32")]
// pub use wasm::{Client, Settings};
