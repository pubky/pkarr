//! Main Crate Error
use ed25519_dalek::SignatureError;
use simple_dns::SimpleDnsError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// For starter, to remove as code matures.
    #[error("Generic error: {0}")]
    Generic(String),
    /// For starter, to remove as code matures.
    #[error("Static error: {0}")]
    Static(&'static str),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[error(transparent)]
    DnsError(#[from] simple_dns::SimpleDnsError),

    #[error(transparent)]
    SignatureError(#[from] SignatureError),

    #[error("Invalid relay payload expected 64 'sig' bytes got: {0}")]
    RelayPayloadInvalidSignatureLength(usize),
    #[error("Invalid relay payload expected 8 'seq' bytes, got: {0}")]
    RelayPayloadInvalidSequenceLength(usize),

    #[error("All attempts to publish failed")]
    PublishFailed,
}
