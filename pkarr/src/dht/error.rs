use mainline::errors::PutMutableError;

use super::ResolveReport;

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

/// DHT resolve error.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolveError {
    /// DHT query did not query any nodes.
    #[error("no DHT nodes were queried")]
    NoNodesQueried,

    /// DHT query did not receive any responses.
    #[error("no queried DHT nodes responded")]
    NoNodesResponded,

    /// DHT query did not receive any valid responses.
    #[error("no responded DHT nodes returned valid response")]
    NoValidResponses,

    /// No signed packet was found.
    #[error("signed packet not found")]
    NotFound,
}

impl From<ResolveReport> for ResolveError {
    fn from(report: ResolveReport) -> Self {
        if report.queried() == 0 {
            Self::NoNodesQueried
        } else if report.responded() == 0 {
            Self::NoNodesResponded
        } else if report.valid_responses() == 0 {
            Self::NoValidResponses
        } else {
            Self::NotFound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(outcome: mainline::GetMutableOutcome) -> ResolveReport {
        ResolveReport::with_invalid_signed_packets(outcome, 0)
    }

    #[test]
    fn resolve_report_without_queried_nodes_maps_to_no_nodes_queried() {
        let error = ResolveError::from(report(mainline::GetMutableOutcome::default()));

        assert_eq!(error, ResolveError::NoNodesQueried);
    }

    #[test]
    fn resolve_report_without_responses_maps_to_no_nodes_responded() {
        let error = ResolveError::from(report(mainline::GetMutableOutcome {
            queried: 20,
            ..Default::default()
        }));

        assert_eq!(error, ResolveError::NoNodesResponded);
    }

    #[test]
    fn resolve_report_without_valid_responses_maps_to_no_valid_responses() {
        let error = ResolveError::from(report(mainline::GetMutableOutcome {
            queried: 20,
            invalid_responses: 1,
            ..Default::default()
        }));

        assert_eq!(error, ResolveError::NoValidResponses);
    }

    #[test]
    fn resolve_report_with_valid_responses_maps_to_not_found() {
        let error = ResolveError::from(report(mainline::GetMutableOutcome {
            queried: 20,
            no_values: 1,
            ..Default::default()
        }));

        assert_eq!(error, ResolveError::NotFound);
    }
}
