use super::resolve_report::ResolveReport;

/// Default minimum number of DHT nodes expected to acknowledge storing a published packet.
const MINIMUM_PUBLISH_STORED_NODES: u32 = 10;

/// Default minimum number of DHT nodes a resolve query should visit.
const DEFAULT_MINIMUM_QUERIED_NODES: u32 = 20;

/// Default minimum number of DHT nodes that should respond to a resolve query.
const DEFAULT_MINIMUM_RESPONDED_NODES: u32 = 3;

/// Default minimum number of valid responses expected from a resolve query.
const DEFAULT_MINIMUM_VALID_RESPONSES: u32 = 1;

/// Testnet minimum number of DHT nodes expected to acknowledge storing a published packet.
const TESTNET_MINIMUM_PUBLISH_STORED_NODES: u32 = 1;

/// Testnet minimum number of DHT nodes a resolve query should visit.
const TESTNET_MINIMUM_QUERIED_NODES: u32 = 1;

/// Testnet minimum number of DHT nodes that should respond to a resolve query.
const TESTNET_MINIMUM_RESPONDED_NODES: u32 = 1;

/// Testnet minimum number of valid responses expected from a resolve query.
const TESTNET_MINIMUM_VALID_RESPONSES: u32 = 1;

/// Policy used to classify DHT query diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReportPolicy {
    /// Minimum number of DHT nodes expected to acknowledge storing a published packet.
    pub minimum_publish_stored_nodes: u32,
    /// Minimum number of unique DHT nodes that should be queried.
    pub minimum_queried_nodes: u32,
    /// Minimum number of queried DHT nodes that should respond.
    pub minimum_responded_nodes: u32,
    /// Minimum number of responses with a valid mutable GET shape.
    pub minimum_valid_responses: u32,
    /// Whether invalid values or invalid response shapes should be reported as warnings.
    pub report_invalid_responses: bool,
    /// Whether KRPC error responses should be reported as warnings.
    pub report_krpc_errors: bool,
}

impl ReportPolicy {
    /// Create a policy with thresholds suitable for the public Mainline DHT.
    pub fn mainnet() -> Self {
        Self {
            minimum_publish_stored_nodes: MINIMUM_PUBLISH_STORED_NODES,
            minimum_queried_nodes: DEFAULT_MINIMUM_QUERIED_NODES,
            minimum_responded_nodes: DEFAULT_MINIMUM_RESPONDED_NODES,
            minimum_valid_responses: DEFAULT_MINIMUM_VALID_RESPONSES,
            report_invalid_responses: true,
            report_krpc_errors: true,
        }
    }

    /// Create a policy with thresholds suitable for small local testnets.
    pub fn testnet() -> Self {
        Self {
            minimum_publish_stored_nodes: TESTNET_MINIMUM_PUBLISH_STORED_NODES,
            minimum_queried_nodes: TESTNET_MINIMUM_QUERIED_NODES,
            minimum_responded_nodes: TESTNET_MINIMUM_RESPONDED_NODES,
            minimum_valid_responses: TESTNET_MINIMUM_VALID_RESPONSES,
            report_invalid_responses: true,
            report_krpc_errors: true,
        }
    }

    /// Classify warning diagnostics for a DHT publish query result.
    pub fn classify_publish_result(&self, stored_at: u32) -> Vec<PublishWarning> {
        if stored_at < self.minimum_publish_stored_nodes {
            vec![PublishWarning::TooFewNodesStored {
                stored_at,
                minimum: self.minimum_publish_stored_nodes,
            }]
        } else {
            Vec::new()
        }
    }

    /// Classify warning diagnostics for a DHT resolve query.
    pub fn classify_resolve_report(&self, report: &ResolveReport) -> Vec<ResolveWarning> {
        let outcome = &report.outcome;
        let invalid_signed_packet_count = report.invalid_signed_packet_count;

        let mut warnings = Vec::new();
        let responded = outcome.responded();
        let valid_responses = outcome
            .valid_responses()
            .saturating_sub(invalid_signed_packet_count);

        if outcome.queried < self.minimum_queried_nodes {
            warnings.push(ResolveWarning::TooFewNodesQueried {
                queried: outcome.queried,
                minimum: self.minimum_queried_nodes,
            });
        }

        if responded < self.minimum_responded_nodes {
            warnings.push(ResolveWarning::TooFewNodesResponded {
                responded,
                minimum: self.minimum_responded_nodes,
            });
        }

        if valid_responses < self.minimum_valid_responses {
            warnings.push(ResolveWarning::TooFewValidResponses {
                valid_responses,
                minimum: self.minimum_valid_responses,
            });
        }

        if self.report_invalid_responses
            && (outcome.invalid_values > 0
                || outcome.invalid_responses > 0
                || invalid_signed_packet_count > 0)
        {
            warnings.push(ResolveWarning::InvalidResponses {
                invalid_values: outcome.invalid_values,
                invalid_responses: outcome.invalid_responses,
                invalid_signed_packets: invalid_signed_packet_count,
            });
        }

        if self.report_krpc_errors && outcome.krpc_errors > 0 {
            warnings.push(ResolveWarning::KrpcErrors {
                krpc_errors: outcome.krpc_errors,
            });
        }

        warnings
    }
}

/// Warning diagnostics found in a DHT publish query.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublishWarning {
    /// Fewer DHT nodes acknowledged storing the packet than expected.
    TooFewNodesStored {
        /// Number of DHT nodes that acknowledged storing the packet.
        stored_at: u32,
        /// Minimum number of storing nodes expected by the policy.
        minimum: u32,
    },
}

/// Warning diagnostics found in a DHT resolve query.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveWarning {
    /// The query visited fewer unique DHT nodes than expected.
    TooFewNodesQueried {
        /// Number of unique DHT nodes queried.
        queried: u32,
        /// Minimum number of queried nodes expected by the policy.
        minimum: u32,
    },
    /// The query received responses from fewer DHT nodes than expected.
    TooFewNodesResponded {
        /// Number of queried DHT nodes that responded.
        responded: u32,
        /// Minimum number of responses expected by the policy.
        minimum: u32,
    },
    /// The query received fewer valid mutable GET responses than expected.
    TooFewValidResponses {
        /// Number of valid mutable GET responses.
        valid_responses: u32,
        /// Minimum number of valid responses expected by the policy.
        minimum: u32,
    },
    /// The query received invalid values or invalid response shapes.
    InvalidResponses {
        /// Number of mutable value responses that failed validation.
        invalid_values: u32,
        /// Number of invalid response shapes returned.
        invalid_responses: u32,
        /// Number of mutable values that failed signed packet validation.
        invalid_signed_packets: u32,
    },
    /// The query received KRPC error responses.
    KrpcErrors {
        /// Number of KRPC error responses returned.
        krpc_errors: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::{PublishWarning, ReportPolicy, MINIMUM_PUBLISH_STORED_NODES};

    #[test]
    fn mainnet_publish_policy_warns_when_too_few_nodes_store_packet() {
        let policy = ReportPolicy::mainnet();

        assert_eq!(
            policy.classify_publish_result(MINIMUM_PUBLISH_STORED_NODES - 1),
            vec![PublishWarning::TooFewNodesStored {
                stored_at: MINIMUM_PUBLISH_STORED_NODES - 1,
                minimum: MINIMUM_PUBLISH_STORED_NODES,
            }]
        );
        assert!(policy
            .classify_publish_result(MINIMUM_PUBLISH_STORED_NODES)
            .is_empty());
    }

    #[test]
    fn testnet_publish_policy_relaxes_stored_nodes_threshold() {
        assert!(ReportPolicy::testnet()
            .classify_publish_result(1)
            .is_empty());
    }
}
