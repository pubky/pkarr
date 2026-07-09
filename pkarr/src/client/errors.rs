#[cfg(relays)]
use crate::relay_client::RelayError;

/// Errors occurring during building a [Client][super::Client]
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    /// Client configured without DHT client or relays.
    #[error("client configured without DHT client or relays")]
    NoNetwork,

    #[cfg(dht)]
    /// Failed to build the Dht client.
    #[error("failed to build the DHT client: {0}")]
    DhtBuildError(std::io::Error),

    #[cfg(relays)]
    /// Passed an empty list of relays.
    #[error("passed an empty list of relays")]
    EmptyListOfRelays,

    #[cfg(relays)]
    /// Failed to build the relays client.
    #[error("failed to build the relays client: {0}")]
    RelayBuildError(RelayError),
}

/// Errors occurring during publishing a [SignedPacket][crate::SignedPacket]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PublishError {
    /// DHT publish had no candidate nodes to query.
    #[error("DHT publish queried no nodes")]
    NoDhtNodesQueried,

    /// No DHT node or relay produced a response for the publish query.
    #[error("publish query received no responses")]
    NoResponses,

    /// DHT publish query was rejected with an explicit error response.
    #[error("DHT publish query was rejected with code: {code}, description: {description}")]
    Rejected {
        /// Error response code.
        code: i32,
        /// Error response description.
        description: String,
    },

    /// A different [SignedPacket][crate::SignedPacket] is being concurrently
    /// published for the same [PublicKey][crate::PublicKey].
    #[error(transparent)]
    Concurrency(#[from] ConcurrencyError),

    /// All responses were unexpected, check debug logs.
    #[error("all responses were unexpected, check debug logs")]
    UnexpectedResponses,
}

/// Errors that require resolving most recent
/// [SignedPacket][crate::SignedPacket] before publishing.
///
/// These errors mean a publish could overwrite a newer packet. Resolve the
/// latest [SignedPacket][crate::SignedPacket] before publishing again.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConcurrencyError {
    /// A different [SignedPacket][crate::SignedPacket] is being concurrently
    /// published for the same [PublicKey][crate::PublicKey].
    #[error("a different SignedPacket is being concurrently published for the same PublicKey")]
    ConflictRisk,

    /// Found a more recent SignedPacket in the client's cache.
    #[error("found a more recent SignedPacket in the client's cache")]
    NotMostRecent,

    /// Compare and swap failed; there is a more recent SignedPacket than the
    /// one seen before publishing.
    #[error(
        "compare and swap failed; there is a more recent SignedPacket than the one seen before publishing"
    )]
    CasFailed,
}

/// Errors occurring during resolving a [SignedPacket][crate::SignedPacket].
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolveError {
    /// DHT resolve queried no nodes.
    #[error("DHT resolve queried no nodes")]
    NoDhtNodesQueried,

    /// No DHT node or relay produced a response for the resolve query.
    #[error("resolve query received no responses")]
    NoResponses,

    /// The resolve query received responses, but none were usable.
    #[error("resolve query received no usable responses")]
    NoUsableResponses,

    /// No [SignedPacket][crate::SignedPacket] was found for the requested
    /// [PublicKey][crate::PublicKey].
    #[error("SignedPacket was not found")]
    NotFound,

    /// The network found a newer mutable item that is not a valid signed packet.
    #[error("mutable item at seq {seq} is not a valid signed packet")]
    InvalidSignedPacket {
        /// Mutable item sequence number.
        seq: i64,
    },

    /// All responses were unexpected, check debug logs.
    #[error("all responses were unexpected, check debug logs")]
    UnexpectedResponses,
}

#[cfg(dht)]
impl From<crate::dht::PublishError> for PublishError {
    fn from(value: crate::dht::PublishError) -> Self {
        match value {
            crate::dht::PublishError::Timeout => PublishError::NoResponses,
            crate::dht::PublishError::NoClosestNodes => PublishError::NoDhtNodesQueried,
            crate::dht::PublishError::ErrorResponse { code, description } => {
                PublishError::Rejected { code, description }
            }
            crate::dht::PublishError::ConflictRisk => {
                PublishError::Concurrency(ConcurrencyError::ConflictRisk)
            }
            crate::dht::PublishError::NotMostRecent => {
                PublishError::Concurrency(ConcurrencyError::NotMostRecent)
            }
            crate::dht::PublishError::CasFailed => {
                PublishError::Concurrency(ConcurrencyError::CasFailed)
            }
        }
    }
}

#[cfg(relays)]
impl From<RelayError> for PublishError {
    fn from(value: RelayError) -> Self {
        match value {
            RelayError::Timeout | RelayError::DhtUnavailable => PublishError::NoResponses,
            RelayError::NotMostRecent => PublishError::Concurrency(ConcurrencyError::NotMostRecent),
            RelayError::CasFailed => PublishError::Concurrency(ConcurrencyError::CasFailed),
            RelayError::ConflictRisk => PublishError::Concurrency(ConcurrencyError::ConflictRisk),
            _ => PublishError::UnexpectedResponses,
        }
    }
}

#[cfg(dht)]
impl From<crate::dht::ResolveError> for ResolveError {
    fn from(value: crate::dht::ResolveError) -> Self {
        match value {
            crate::dht::ResolveError::NoNodesQueried => ResolveError::NoDhtNodesQueried,
            crate::dht::ResolveError::NoNodesResponded => ResolveError::NoResponses,
            crate::dht::ResolveError::NoUsableResponses => ResolveError::NoUsableResponses,
            crate::dht::ResolveError::InvalidSignedPacket { seq } => {
                ResolveError::InvalidSignedPacket { seq }
            }
            crate::dht::ResolveError::NotFound => ResolveError::NotFound,
        }
    }
}

#[cfg(relays)]
impl From<RelayError> for ResolveError {
    fn from(value: RelayError) -> Self {
        match value {
            RelayError::Timeout | RelayError::DhtUnavailable => ResolveError::NoResponses,
            RelayError::NotFound => ResolveError::NotFound,
            RelayError::InvalidSignedPacketSeq { seq } => ResolveError::InvalidSignedPacket { seq },
            _ => ResolveError::UnexpectedResponses,
        }
    }
}

#[cfg(all(test, relays))]
mod relay_tests {
    use super::*;
    use crate::relay_client::RelayError;

    #[test]
    fn dht_relay_errors_map_to_publish_errors() {
        assert!(matches!(
            PublishError::from(RelayError::DhtUnavailable),
            PublishError::NoResponses
        ));
    }

    #[test]
    fn dht_relay_errors_map_to_resolve_errors() {
        assert!(matches!(
            ResolveError::from(RelayError::DhtUnavailable),
            ResolveError::NoResponses
        ));
    }

    #[test]
    fn relay_timeouts_map_to_operation_specific_errors() {
        assert_eq!(
            PublishError::from(RelayError::Timeout),
            PublishError::NoResponses
        );
        assert_eq!(
            ResolveError::from(RelayError::Timeout),
            ResolveError::NoResponses
        );
    }

    #[test]
    fn relay_bad_request_is_an_unexpected_resolve_response() {
        assert_eq!(
            ResolveError::from(RelayError::BadRequest),
            ResolveError::UnexpectedResponses
        );
    }

    #[test]
    fn relay_bad_request_is_an_unexpected_publish_response() {
        assert_eq!(
            PublishError::from(RelayError::BadRequest),
            PublishError::UnexpectedResponses
        );
    }
}

#[cfg(all(test, dht))]
mod dht_tests {
    use super::*;

    #[test]
    fn dht_publish_without_closest_nodes_maps_to_no_dht_nodes_queried() {
        assert_eq!(
            PublishError::from(crate::dht::PublishError::NoClosestNodes),
            PublishError::NoDhtNodesQueried
        );
    }

    #[test]
    fn dht_resolve_without_responses_is_not_a_generic_query_timeout() {
        assert_eq!(
            ResolveError::from(crate::dht::ResolveError::NoNodesResponded),
            ResolveError::NoResponses
        );
    }

    #[test]
    fn dht_resolve_without_usable_responses_is_preserved() {
        assert_eq!(
            ResolveError::from(crate::dht::ResolveError::NoUsableResponses),
            ResolveError::NoUsableResponses
        );
    }
}
