#[derive(thiserror::Error, Debug)]
/// Errors occurring during building a [Client][super::Client]
pub enum BuildError {
    #[error("Client configured without Mainline node or relays.")]
    /// Client configured without Mainline node or relays.
    NoNetwork,

    #[error("Failed to build the Dht client {0}")]
    /// Failed to build the Dht client.
    DhtBuildError(std::io::Error),

    #[cfg(relays)]
    #[error("Failed to build the relays client {0}")]
    /// Failed to build the relays client.
    RelayBuildError(crate::relay_client::RelayError),

    #[error("Passed an empty list of relays")]
    /// Passed an empty list of relays
    EmptyListOfRelays,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
/// Errors occurring during publishing a [SignedPacket][crate::SignedPacket]
pub enum PublishError {
    #[error(transparent)]
    /// Errors that requires either a retry or debugging the network condition.
    Query(#[from] QueryError),

    #[error(transparent)]
    /// A different [SignedPacket][crate::SignedPacket] is being concurrently published for the same [PublicKey][crate::PublicKey].
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket][crate::SignedPacket] before publishing again.
    Concurrency(#[from] ConcurrencyError),

    // === Relays only errors ===
    #[error("All relays responded with unexpected responses, check debug logs.")]
    /// All relays responded with unexpected responses, check debug logs.
    UnexpectedResponses,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
/// Errors occurring during resolving a [SignedPacket][crate::SignedPacket].
pub enum ResolveError {
    #[error(transparent)]
    /// Errors that require either a retry or debugging the network condition.
    Query(#[from] QueryError),

    #[error("DHT resolve queried no nodes")]
    /// DHT resolve queried no nodes.
    NoDhtNodesQueried,

    #[error("DHT resolve found no valid responses")]
    /// DHT resolve received responses, but none contained valid mutable records.
    NoValidDhtResponses,

    #[error("SignedPacket was not found")]
    /// No [SignedPacket][crate::SignedPacket] was found for the requested [PublicKey][crate::PublicKey].
    NotFound,

    #[error("DHT mutable item at seq {seq} is not a valid signed packet")]
    /// DHT found a newer mutable item that is not a valid signed packet.
    InvalidSignedPacket {
        /// Mutable item sequence number.
        seq: i64,
    },

    #[error("All relays responded with unexpected responses, check debug logs.")]
    /// All relays responded with unexpected responses, check debug logs.
    UnexpectedResponses,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
/// Errors that requires either a retry or debugging the network condition.
pub enum QueryError {
    /// Publish query timed out with no responses neither success or errors, from Dht or relays.
    #[error("Publish query timed out with no responses neither success or errors.")]
    Timeout,

    #[error("Publishing SignedPacket to Mainline failed.")]
    /// Publishing SignedPacket to Mainline failed.
    NoClosestNodes,

    #[error("Publishing SignedPacket to Mainline failed code: {0}, description: {1}.")]
    /// Publishing SignedPacket to Mainline failed, received an error response.
    DhtErrorResponse(i32, String),

    #[error("Most relays responded with bad request")]
    /// Most relays responded with bad request
    BadRequest,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
/// Errors that requires resolving most recent [SignedPacket][crate::SignedPacket] before publishing.
pub enum ConcurrencyError {
    #[error("A different SignedPacket is being concurrently published for the same PublicKey.")]
    /// A different [SignedPacket][crate::SignedPacket] is being concurrently published for the same [PublicKey][crate::PublicKey].
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket][crate::SignedPacket] before publishing again.
    ConflictRisk,

    #[error("Found a more recent SignedPacket in the client's cache")]
    /// Found a more recent SignedPacket in the client's cache
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket][crate::SignedPacket] before publishing again.
    NotMostRecent,

    #[error("Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing")]
    /// Compare and swap failed; there is a more recent SignedPacket than the one seen before publishing
    ///
    /// This risks a lost update, you should resolve most recent [SignedPacket][crate::SignedPacket] before publishing again.
    CasFailed,
}

#[cfg(dht)]
impl From<crate::dht::PublishError> for PublishError {
    fn from(value: crate::dht::PublishError) -> Self {
        match value {
            crate::dht::PublishError::Timeout => PublishError::Query(QueryError::Timeout),
            crate::dht::PublishError::NoClosestNodes => {
                PublishError::Query(QueryError::NoClosestNodes)
            }
            crate::dht::PublishError::ErrorResponse { code, description } => {
                PublishError::Query(QueryError::DhtErrorResponse(code, description))
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
impl From<crate::relay_client::RelayError> for PublishError {
    fn from(value: crate::relay_client::RelayError) -> Self {
        if let Some(error) = relay_query_error(&value) {
            return PublishError::Query(error);
        }

        if let Some(error) = relay_concurrency_error(&value) {
            return PublishError::Concurrency(error);
        }

        PublishError::UnexpectedResponses
    }
}

#[cfg(dht)]
impl From<crate::dht::ResolveReport> for ResolveError {
    fn from(report: crate::dht::ResolveReport) -> Self {
        if report.queried() == 0 {
            ResolveError::NoDhtNodesQueried
        } else if report.responded() == 0 {
            ResolveError::Query(QueryError::Timeout)
        } else if report.usable_responses() == 0 {
            ResolveError::NoValidDhtResponses
        } else {
            ResolveError::NotFound
        }
    }
}

#[cfg(dht)]
impl From<crate::dht::ResolveError> for ResolveError {
    fn from(value: crate::dht::ResolveError) -> Self {
        match value {
            crate::dht::ResolveError::NoNodesQueried => ResolveError::NoDhtNodesQueried,
            crate::dht::ResolveError::NoNodesResponded => ResolveError::Query(QueryError::Timeout),
            crate::dht::ResolveError::NoUsableResponses => ResolveError::NoValidDhtResponses,
            crate::dht::ResolveError::InvalidSignedPacket { seq } => {
                ResolveError::InvalidSignedPacket { seq }
            }
            crate::dht::ResolveError::NotFound => ResolveError::NotFound,
        }
    }
}

#[cfg(relays)]
impl From<crate::relay_client::RelayError> for ResolveError {
    fn from(value: crate::relay_client::RelayError) -> Self {
        if let Some(error) = relay_query_error(&value) {
            return ResolveError::Query(error);
        }

        match value {
            crate::relay_client::RelayError::NotFound => ResolveError::NotFound,
            crate::relay_client::RelayError::InvalidSignedPacketSeq { seq } => {
                ResolveError::InvalidSignedPacket { seq }
            }
            _ => ResolveError::UnexpectedResponses,
        }
    }
}

#[cfg(relays)]
fn relay_query_error(error: &crate::relay_client::RelayError) -> Option<QueryError> {
    match error {
        crate::relay_client::RelayError::Timeout
        | crate::relay_client::RelayError::DhtUnavailable => Some(QueryError::Timeout),
        crate::relay_client::RelayError::BadRequest => Some(QueryError::BadRequest),
        _ => None,
    }
}

#[cfg(relays)]
fn relay_concurrency_error(error: &crate::relay_client::RelayError) -> Option<ConcurrencyError> {
    match error {
        crate::relay_client::RelayError::NotMostRecent => Some(ConcurrencyError::NotMostRecent),
        crate::relay_client::RelayError::CasFailed => Some(ConcurrencyError::CasFailed),
        crate::relay_client::RelayError::ConflictRisk => Some(ConcurrencyError::ConflictRisk),
        _ => None,
    }
}

#[cfg(all(test, relays))]
mod tests {
    use super::*;
    use crate::relay_client::RelayError;

    #[test]
    fn dht_relay_errors_map_to_publish_errors() {
        assert!(matches!(
            PublishError::from(RelayError::DhtUnavailable),
            PublishError::Query(QueryError::Timeout)
        ));
    }

    #[test]
    fn dht_relay_errors_map_to_resolve_errors() {
        assert!(matches!(
            ResolveError::from(RelayError::DhtUnavailable),
            ResolveError::Query(QueryError::Timeout)
        ));
    }
}
