use crate::SignedPacket;

/// Successful immediate DHT resolve result.
pub struct ResolveFound<F> {
    /// First valid signed packet returned by the DHT query.
    pub first: SignedPacket,
    /// Future that drains the rest of the query and returns the most recent packet and report.
    pub completion: F,
}

/// Default minimum number of DHT nodes a resolve query should visit.
const DEFAULT_MINIMUM_QUERIED_NODES: u32 = 20;

/// Default minimum number of DHT nodes that should respond to a resolve query.
const DEFAULT_MINIMUM_RESPONDED_NODES: u32 = 3;

/// Default minimum number of valid responses expected from a resolve query.
const DEFAULT_MINIMUM_VALID_RESPONSES: u32 = 1;

/// Policy used to classify DHT resolve query diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolveReportPolicy {
    /// Minimum number of unique DHT nodes that should be queried.
    minimum_queried_nodes: u32,
    /// Minimum number of queried DHT nodes that should respond.
    minimum_responded_nodes: u32,
    /// Minimum number of responses with a valid mutable GET shape.
    minimum_valid_responses: u32,
    /// Whether invalid values or invalid response shapes should be reported as suspicious.
    flag_invalid_responses: bool,
    /// Whether KRPC error responses should be reported as suspicious.
    flag_krpc_errors: bool,
}

impl Default for ResolveReportPolicy {
    fn default() -> Self {
        Self {
            minimum_queried_nodes: DEFAULT_MINIMUM_QUERIED_NODES,
            minimum_responded_nodes: DEFAULT_MINIMUM_RESPONDED_NODES,
            minimum_valid_responses: DEFAULT_MINIMUM_VALID_RESPONSES,
            flag_invalid_responses: true,
            flag_krpc_errors: true,
        }
    }
}

impl ResolveReportPolicy {
    /// Classify a raw Mainline mutable GET outcome with invalid signed packet diagnostics.
    fn classify_with_invalid_signed_packets(
        &self,
        outcome: mainline::GetMutableOutcome,
        invalid_signed_packet_count: u32,
    ) -> ResolveReport {
        ResolveReport {
            suspicions: self.suspicions(&outcome, invalid_signed_packet_count),
            outcome,
            invalid_signed_packet_count,
        }
    }

    fn suspicions(
        &self,
        outcome: &mainline::GetMutableOutcome,
        invalid_signed_packet_count: u32,
    ) -> Vec<ResolveSuspicion> {
        let mut suspicions = Vec::new();
        let responded = outcome.responded();
        let valid_responses = outcome
            .valid_responses()
            .saturating_sub(invalid_signed_packet_count);

        if outcome.queried < self.minimum_queried_nodes {
            suspicions.push(ResolveSuspicion::TooFewNodesQueried {
                queried: outcome.queried,
                minimum: self.minimum_queried_nodes,
            });
        }

        if responded < self.minimum_responded_nodes {
            suspicions.push(ResolveSuspicion::TooFewNodesResponded {
                responded,
                minimum: self.minimum_responded_nodes,
            });
        }

        if valid_responses < self.minimum_valid_responses {
            suspicions.push(ResolveSuspicion::TooFewValidResponses {
                valid_responses,
                minimum: self.minimum_valid_responses,
            });
        }

        if self.flag_invalid_responses
            && (outcome.invalid_values > 0
                || outcome.invalid_responses > 0
                || invalid_signed_packet_count > 0)
        {
            suspicions.push(ResolveSuspicion::InvalidResponses {
                invalid_values: outcome.invalid_values,
                invalid_responses: outcome.invalid_responses,
                invalid_signed_packets: invalid_signed_packet_count,
            });
        }

        if self.flag_krpc_errors && outcome.krpc_errors > 0 {
            suspicions.push(ResolveSuspicion::KrpcErrors {
                krpc_errors: outcome.krpc_errors,
            });
        }

        suspicions
    }
}

/// Classified diagnostics for a DHT resolve query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveReport {
    outcome: mainline::GetMutableOutcome,
    invalid_signed_packet_count: u32,
    suspicions: Vec<ResolveSuspicion>,
}

impl ResolveReport {
    /// Classify a raw Mainline mutable GET outcome with invalid signed packet diagnostics.
    pub(crate) fn with_invalid_signed_packets(
        outcome: mainline::GetMutableOutcome,
        invalid_signed_packet_count: u32,
    ) -> Self {
        ResolveReportPolicy::default()
            .classify_with_invalid_signed_packets(outcome, invalid_signed_packet_count)
    }

    /// Returns true when the resolve query had suspicious diagnostics.
    pub fn is_suspicious(&self) -> bool {
        !self.suspicions.is_empty()
    }

    /// Number of unique DHT nodes queried.
    pub fn queried(&self) -> u32 {
        self.outcome.queried
    }

    /// Number of nodes that returned a GET response before timing out.
    pub fn responded(&self) -> u32 {
        self.outcome.responded()
    }

    /// Number of valid responses after deducting invalid signed packets.
    pub fn valid_responses(&self) -> u32 {
        self.outcome
            .valid_responses()
            .saturating_sub(self.invalid_signed_packet_count)
    }

    /// Suspicious diagnostics found for this resolve query.
    pub fn suspicions(&self) -> &[ResolveSuspicion] {
        &self.suspicions
    }
}

/// Suspicious diagnostics found in a DHT resolve query.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveSuspicion {
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
    use super::*;

    #[test]
    fn healthy_outcome_is_ok() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 20,
                values: 1,
                no_values: 2,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            0,
        );

        assert!(!report.is_suspicious());
        assert!(report.suspicions().is_empty());
    }

    #[test]
    fn sparse_outcome_is_suspicious() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 2,
                values: 0,
                no_values: 0,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            0,
        );

        assert!(report.is_suspicious());
        assert_eq!(
            report.suspicions(),
            &[
                ResolveSuspicion::TooFewNodesQueried {
                    queried: 2,
                    minimum: DEFAULT_MINIMUM_QUERIED_NODES,
                },
                ResolveSuspicion::TooFewNodesResponded {
                    responded: 0,
                    minimum: DEFAULT_MINIMUM_RESPONDED_NODES,
                },
                ResolveSuspicion::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: DEFAULT_MINIMUM_VALID_RESPONSES,
                },
            ]
        );
    }

    #[test]
    fn invalid_and_error_responses_are_suspicious() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 20,
                values: 1,
                no_values: 2,
                no_more_recent: 0,
                invalid_values: 1,
                invalid_responses: 1,
                krpc_errors: 1,
            },
            0,
        );

        assert!(report.is_suspicious());
        assert_eq!(
            report.suspicions(),
            &[
                ResolveSuspicion::InvalidResponses {
                    invalid_values: 1,
                    invalid_responses: 1,
                    invalid_signed_packets: 0,
                },
                ResolveSuspicion::KrpcErrors { krpc_errors: 1 },
            ]
        );
    }

    #[test]
    fn invalid_signed_packets_are_suspicious() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 20,
                values: 3,
                no_values: 2,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            2,
        );

        assert!(report.is_suspicious());
        assert_eq!(
            report.suspicions(),
            &[ResolveSuspicion::InvalidResponses {
                invalid_values: 0,
                invalid_responses: 0,
                invalid_signed_packets: 2,
            }]
        );
    }

    #[test]
    fn invalid_signed_packets_are_deducted_from_valid_responses() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 20,
                values: 3,
                no_values: 0,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            3,
        );

        assert!(report.is_suspicious());
        assert_eq!(report.valid_responses(), 0);
        assert_eq!(
            report.suspicions(),
            &[
                ResolveSuspicion::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: DEFAULT_MINIMUM_VALID_RESPONSES,
                },
                ResolveSuspicion::InvalidResponses {
                    invalid_values: 0,
                    invalid_responses: 0,
                    invalid_signed_packets: 3,
                },
            ]
        );
    }

    #[test]
    fn invalid_signed_packet_deduction_saturates_valid_responses() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 20,
                values: 1,
                no_values: 0,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            2,
        );

        assert_eq!(report.valid_responses(), 0);
        assert_eq!(
            report.suspicions(),
            &[
                ResolveSuspicion::TooFewNodesResponded {
                    responded: 1,
                    minimum: DEFAULT_MINIMUM_RESPONDED_NODES,
                },
                ResolveSuspicion::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: DEFAULT_MINIMUM_VALID_RESPONSES,
                },
                ResolveSuspicion::InvalidResponses {
                    invalid_values: 0,
                    invalid_responses: 0,
                    invalid_signed_packets: 2,
                },
            ]
        );
    }
}
