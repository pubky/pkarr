use crate::PublicKey;

/// Resolution failure. No error authorizes weaker TLS.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The packet source failed. The original adapter error is retained.
    #[error("packet provider failed for {key}: {source}")]
    Provider {
        /// The requested owner.
        key: Box<PublicKey>,
        /// Network, cache, or storage failure from the provider.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The input is not an ASCII DNS name ending in a canonical Pubky key.
    #[error("invalid Pubky service name: {0}")]
    InvalidServiceName(String),
    /// The request has no supported protocols or an invalid port.
    #[error("invalid resolution request: {0}")]
    InvalidRequest(&'static str),
    /// The caller supplied invalid clocks or an unrepresentable deadline.
    #[error("invalid resolution time: {0}")]
    InvalidTime(&'static str),
    /// The provider has no packet for the requested service.
    #[error("missing packet for {0}")]
    MissingPacket(Box<PublicKey>),
    /// A consulted record is malformed or ambiguous.
    #[error("invalid record: {0}")]
    InvalidRecord(&'static str),
    /// A required record's observation-based TTL has expired.
    #[error("expired record")]
    ExpiredRecord,
    /// The requested protocol feature is outside this implementation's scope.
    #[error("unsupported resolution feature: {0}")]
    Unsupported(&'static str),
    /// A signed HTTPS AliasMode record with target root disables the service.
    #[error("service unavailable")]
    ServiceUnavailable,
    /// No compatible ServiceMode record was found. No address fallback applies.
    #[error("no compatible endpoint")]
    NoCompatibleEndpoint,
    /// No compatible service has usable signed addresses or address hints.
    #[error("no addresses")]
    NoAddresses,
}
