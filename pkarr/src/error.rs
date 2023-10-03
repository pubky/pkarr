//! Main Crate Error
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
    SignatureError(#[from] ed25519_dalek::SignatureError),

    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),

    #[error("Invalid SignedPacket bytes length, expected at least 72 bytes but got: {0}")]
    InvalidSingedPacketBytes(usize),

    #[error(
        "Encoded and compressed DNS Packet is too large, expected max 1000 bytes but got: {0}"
    )]
    PacketTooLarge(usize),

    #[error("All attempts to publish failed")]
    PublishFailed,
}
