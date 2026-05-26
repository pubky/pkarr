use mainline::errors::PutMutableError;

/// DHT publish error.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PublishError {
    /// DHT query timed out.
    #[error("DHT query timed out")]
    Timeout,

    /// DHT query found no closest nodes.
    #[error("DHT query found no closest nodes")]
    NoClosestNodes,

    /// DHT query returned an error response.
    #[error("DHT query returned error response code: {code}, description: {description}")]
    ErrorResponse {
        /// Error response code.
        code: i32,
        /// Error response description.
        description: String,
    },

    /// Publishing risks a write conflict.
    #[error("publishing risks a write conflict")]
    ConflictRisk,

    /// Packet is not the most recent.
    #[error("packet is not the most recent")]
    NotMostRecent,

    /// Compare-and-swap failed.
    #[error("compare-and-swap failed")]
    CasFailed,
}

impl From<PutMutableError> for PublishError {
    fn from(value: PutMutableError) -> Self {
        match value {
            PutMutableError::Query(error) => match error {
                mainline::errors::PutQueryError::Timeout => PublishError::Timeout,
                mainline::errors::PutQueryError::NoClosestNodes => PublishError::NoClosestNodes,
                mainline::errors::PutQueryError::ErrorResponse(error) => {
                    PublishError::ErrorResponse {
                        code: error.code,
                        description: error.description,
                    }
                }
            },
            PutMutableError::Concurrency(error) => match error {
                mainline::errors::ConcurrencyError::ConflictRisk => PublishError::ConflictRisk,
                mainline::errors::ConcurrencyError::NotMostRecent => PublishError::NotMostRecent,
                mainline::errors::ConcurrencyError::CasFailed => PublishError::CasFailed,
            },
        }
    }
}
