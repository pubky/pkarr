//! Crate prelude
// Re-export the crate Error.
pub use crate::error::Error;

// Alias Result to be the crate Result.
pub type Result<T> = core::result::Result<T, Error>;

// Generic Wrapper tuple struct for newtype pattern,
// mostly for external type to type From/TryFrom conversions
pub struct W<T>(pub T);
