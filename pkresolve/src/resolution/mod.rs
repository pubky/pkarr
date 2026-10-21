//! Pubky service resolution with synchronous and asynchronous packet providers.

mod https_records;
mod resolve;
mod routing;
mod traversal;
mod types;

pub use resolve::{derive_resolution, derive_resolution_async};
pub use types::{EndpointCandidate, HttpProtocol, Resolution, ResolutionRequest};
