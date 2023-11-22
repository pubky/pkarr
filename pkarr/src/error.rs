//! Main Crate Error

#[derive(thiserror::Error, Debug)]
/// Pkarr crate error enum.
pub enum Error {
    /// For starter, to remove as code matures.
    #[error("Generic error: {0}")]
    Generic(String),
    /// For starter, to remove as code matures.
    #[error("Static error: {0}")]
    Static(&'static str),

    #[error(transparent)]
    /// Transparent [std::io::Error]
    IO(#[from] std::io::Error),

    #[error(transparent)]
    /// Transparent [simple_dns::SimpleDnsError]
    DnsError(#[from] simple_dns::SimpleDnsError),

    #[error(transparent)]
    /// Transparent [ed25519_dalek::SignatureError]
    SignatureError(#[from] ed25519_dalek::SignatureError),

    #[error(transparent)]
    /// Transparent [reqwest::Error]
    ReqwestError(#[from] reqwest::Error),

    #[error("Relay {0} responded with an error: {1} {2}")]
    /// Relay response is not 200 OK
    RelayResponse(url::Url, reqwest::StatusCode, String),

    #[error("Invalid SignedPacket bytes length, expected at least 72 bytes but got: {0}")]
    /// DNS packet failed to decode or encode
    InvalidSingedPacketBytes(usize),

    #[error(
        "Encoded and compressed DNS Packet is too large, expected max 1000 bytes but got: {0}"
    )]
    // DNS packet endocded and compressed is larger than 1000 bytes
    PacketTooLarge(usize),

    #[error("All attempts to publish failed")]
    /// Relay response is not 200 OK
    PublishFailed,

    #[error(transparent)]
    MainlineError(#[from] mainline::Error),
}
