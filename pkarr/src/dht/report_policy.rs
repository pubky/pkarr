use super::resolve_report::ResolveReport;

/// Policy used to classify DHT query diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReportPolicy {
    /// Minimum number of DHT nodes expected to acknowledge storing a published packet.
    pub minimum_publish_stored_nodes: u32,
    /// Minimum number of unique DHT nodes that should be queried.
    pub minimum_queried_nodes: u32,
    /// Minimum number of queried DHT nodes that should respond.
    pub minimum_responded_nodes: u32,
    /// Minimum number of responses with a usable mutable GET shape.
    pub minimum_usable_responses: u32,
    /// Whether unusable mutable GET responses should be reported as
    /// [`ResolveWarning::MalformedResponses`].
    ///
    /// This covers mutable values that fail validation and malformed response
    /// shapes, but not explicit KRPC error responses.
    pub report_malformed_responses: bool,
    /// Whether KRPC error responses should be reported as warnings.
    ///
    /// KRPC errors are also unusable resolve responses, but they are kept
    /// separate from malformed responses because they usually point to a
    /// protocol-level error returned by the remote node.
    pub report_krpc_errors: bool,
}

impl ReportPolicy {
    const MAINNET: Self = Self {
        minimum_publish_stored_nodes: 10,
        minimum_queried_nodes: 20,
        minimum_responded_nodes: 5,
        minimum_usable_responses: 3,
        report_malformed_responses: true,
        report_krpc_errors: true,
    };

    const TESTNET: Self = Self {
        minimum_publish_stored_nodes: 1,
        minimum_queried_nodes: 1,
        minimum_responded_nodes: 1,
        minimum_usable_responses: 1,
        report_malformed_responses: false,
        report_krpc_errors: false,
    };

    /// Create a policy with thresholds suitable for the public Mainline DHT.
    pub fn mainnet() -> Self {
        Self::MAINNET
    }

    /// Create a policy with thresholds suitable for small local testnets.
    pub fn testnet() -> Self {
        Self::TESTNET
    }

    /// Classify warning diagnostics for a DHT publish query result.
    pub fn classify_publish_result(&self, stored_on: u32) -> Vec<PublishWarning> {
        if stored_on < self.minimum_publish_stored_nodes {
            vec![PublishWarning::TooFewNodesStored {
                stored_on,
                minimum: self.minimum_publish_stored_nodes,
            }]
        } else {
            Vec::new()
        }
    }

    /// Classify warning diagnostics for a DHT resolve query.
    pub fn classify_resolve_report(&self, report: &ResolveReport) -> Vec<ResolveWarning> {
        let outcome = &report.0;

        let mut warnings = Vec::new();
        let responded = outcome.responded();
        let usable_responses = outcome.valid_responses();

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

        if usable_responses < self.minimum_usable_responses {
            warnings.push(ResolveWarning::TooFewUsableResponses {
                usable_responses,
                minimum: self.minimum_usable_responses,
            });
        }

        // Values that fail validation and malformed response shapes are grouped
        // together because both are unusable non-error responses from DHT nodes.
        if self.report_malformed_responses
            && (outcome.invalid_values > 0 || outcome.invalid_responses > 0)
        {
            warnings.push(ResolveWarning::MalformedResponses {
                malformed_values: outcome.invalid_values,
                malformed_responses: outcome.invalid_responses,
            });
        }

        // KRPC errors are unusable too, but unlike malformed responses they
        // were explicit protocol-level errors returned by the queried node.
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
        stored_on: u32,
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
    /// The query received fewer usable mutable GET responses than expected.
    TooFewUsableResponses {
        /// Number of usable mutable GET responses.
        usable_responses: u32,
        /// Minimum number of usable responses expected by the policy.
        minimum: u32,
    },
    /// The query received malformed or unusable mutable GET responses.
    ///
    /// These are responses that did not contain an explicit KRPC error, but
    /// still could not be used as usable mutable GET responses.
    MalformedResponses {
        /// Number of mutable value responses that failed validation.
        malformed_values: u32,
        /// Number of malformed response shapes returned.
        malformed_responses: u32,
    },
    /// The query received KRPC error responses.
    ///
    /// These are also unusable for resolution, but are kept separate from
    /// [`ResolveWarning::MalformedResponses`] because the remote node returned an
    /// explicit protocol-level error.
    KrpcErrors {
        /// Number of KRPC error responses returned.
        krpc_errors: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::{PublishWarning, ReportPolicy, ResolveReport, ResolveWarning};

    fn report(outcome: mainline::GetMutableOutcome) -> ResolveReport {
        ResolveReport::new(outcome)
    }

    fn classify(report: &ResolveReport) -> Vec<ResolveWarning> {
        ReportPolicy::mainnet().classify_resolve_report(report)
    }

    #[test]
    fn mainnet_publish_policy_warns_when_too_few_nodes_store_packet() {
        let policy = ReportPolicy::mainnet();
        let minimum = policy.minimum_publish_stored_nodes;

        assert_eq!(
            policy.classify_publish_result(minimum - 1),
            vec![PublishWarning::TooFewNodesStored {
                stored_on: minimum - 1,
                minimum,
            }]
        );
        assert!(policy.classify_publish_result(minimum).is_empty());
    }

    #[test]
    fn testnet_publish_policy_relaxes_stored_nodes_threshold() {
        assert!(ReportPolicy::testnet()
            .classify_publish_result(1)
            .is_empty());
    }

    #[test]
    fn healthy_resolve_outcome_is_ok() {
        let report = report(mainline::GetMutableOutcome {
            queried: 20,
            values: 1,
            no_values: 4,
            no_more_recent: 0,
            invalid_values: 0,
            invalid_responses: 0,
            krpc_errors: 0,
        });

        assert!(classify(&report).is_empty());
    }

    #[test]
    fn sparse_resolve_outcome_produces_warnings() {
        let report = report(mainline::GetMutableOutcome {
            queried: 2,
            values: 0,
            no_values: 0,
            no_more_recent: 0,
            invalid_values: 0,
            invalid_responses: 0,
            krpc_errors: 0,
        });

        let policy = ReportPolicy::mainnet();
        assert_eq!(
            classify(&report),
            vec![
                ResolveWarning::TooFewNodesQueried {
                    queried: 2,
                    minimum: policy.minimum_queried_nodes,
                },
                ResolveWarning::TooFewNodesResponded {
                    responded: 0,
                    minimum: policy.minimum_responded_nodes,
                },
                ResolveWarning::TooFewUsableResponses {
                    usable_responses: 0,
                    minimum: policy.minimum_usable_responses,
                },
            ]
        );
    }

    #[test]
    fn malformed_and_error_resolve_responses_produce_warnings() {
        let report = report(mainline::GetMutableOutcome {
            queried: 20,
            values: 1,
            no_values: 2,
            no_more_recent: 0,
            invalid_values: 1,
            invalid_responses: 1,
            krpc_errors: 1,
        });

        assert_eq!(
            classify(&report),
            vec![
                ResolveWarning::MalformedResponses {
                    malformed_values: 1,
                    malformed_responses: 1,
                },
                ResolveWarning::KrpcErrors { krpc_errors: 1 },
            ]
        );
    }

    #[test]
    fn testnet_resolve_policy_relaxes_topology_thresholds() {
        let report = report(mainline::GetMutableOutcome {
            queried: 2,
            values: 0,
            no_values: 1,
            no_more_recent: 0,
            invalid_values: 0,
            invalid_responses: 0,
            krpc_errors: 0,
        });

        assert!(ReportPolicy::testnet()
            .classify_resolve_report(&report)
            .is_empty());
    }

    #[test]
    fn custom_resolve_policy_controls_warning_thresholds() {
        let report = report(mainline::GetMutableOutcome {
            queried: 2,
            values: 0,
            no_values: 1,
            no_more_recent: 0,
            invalid_values: 1,
            invalid_responses: 0,
            krpc_errors: 1,
        });

        let warnings = ReportPolicy {
            minimum_publish_stored_nodes: 1,
            minimum_queried_nodes: 2,
            minimum_responded_nodes: 1,
            minimum_usable_responses: 1,
            report_malformed_responses: false,
            report_krpc_errors: false,
        }
        .classify_resolve_report(&report);

        assert!(warnings.is_empty());
    }
}
