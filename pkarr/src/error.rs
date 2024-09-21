//! Main Crate Error

// Alias Result to be the crate Result.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(thiserror::Error, Debug)]
/// Pkarr crate error enum.
pub enum Error {
    #[error(transparent)]
    /// Transparent [std::io::Error]
    IO(#[from] std::io::Error),

    #[cfg(all(not(target_arch = "wasm32"), feature = "dht"))]
    #[error(transparent)]
    /// Transparent [mainline::Error]
    MainlineError(#[from] mainline::Error),

    // === Keys errors ===
    #[error("Invalid PublicKey length, expected 32 bytes but got: {0}")]
    InvalidPublicKeyLength(usize),

    #[error("Invalid Ed25519 publickey; Cannot decompress Edwards point")]
    InvalidEd25519PublicKey,

    #[error("Invalid Ed25519 signature")]
    InvalidEd25519Signature,

    #[error("Invalid PublicKey encoding")]
    InvalidPublicKeyEncoding,

    // === Packets errors ===
    #[error(transparent)]
    /// Transparent [simple_dns::SimpleDnsError]
    DnsError(#[from] simple_dns::SimpleDnsError),

    #[error("Invalid SignedPacket bytes length, expected at least 104 bytes but got: {0}")]
    /// Serialized signed packets are `<32 bytes publickey><64 bytes signature><8 bytes
    /// timestamp><less than or equal to 1000 bytes encoded dns packet>`.
    InvalidSignedPacketBytesLength(usize),

    #[error("Invalid relay payload size, expected at least 72 bytes but got: {0}")]
    /// Relay api http-body should be `<64 bytes signature><8 bytes timestamp>
    /// <less than or equal to 1000 bytes encoded dns packet>`.
    InvalidRelayPayloadSize(usize),

    #[error("DNS Packet is too large, expected max 1000 bytes but got: {0}")]
    // DNS packet endocded and compressed is larger than 1000 bytes
    PacketTooLarge(usize),

    // === Flume errors ===
    #[error(transparent)]
    /// Transparent [flume::RecvError]
    Receive(#[from] flume::RecvError),

    #[error("Dht is shutdown")]
    /// The dht was shutdown.
    DhtIsShutdown,

    #[error("Publish query is already inflight for the same public_key")]
    /// [crate::dht::Client::publish] is already inflight to the same public_key
    PublishInflight,

    #[error("SignedPacket's timestamp is not the most recent")]
    /// Failed to publish because there is a more recent packet.
    NotMostRecent,

    // === Relay errors ===
    #[cfg(feature = "relay")]
    #[error(transparent)]
    /// Transparent [reqwest::Error]
    RelayError(#[from] reqwest::Error),

    #[cfg(feature = "relay")]
    #[error("Empty list of relays")]
    /// Empty list of relays
    EmptyListOfRelays,
}
