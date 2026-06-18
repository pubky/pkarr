/// Diagnostics for a DHT resolve query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveReport {
    pub(super) outcome: mainline::GetMutableOutcome,
    pub(super) invalid_signed_packet_count: u32,
}

impl ResolveReport {
    /// Build a report from a raw Mainline mutable GET outcome with invalid signed packet diagnostics.
    pub(crate) fn with_invalid_signed_packets(
        outcome: mainline::GetMutableOutcome,
        invalid_signed_packet_count: u32,
    ) -> Self {
        Self {
            outcome,
            invalid_signed_packet_count,
        }
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
}

#[cfg(test)]
mod tests {
    use super::super::report_policy::{ReportPolicy, ResolveWarning};
    use super::*;

    fn classify(report: &ResolveReport) -> Vec<ResolveWarning> {
        ReportPolicy::mainnet().classify_resolve_report(report)
    }

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

        assert!(classify(&report).is_empty());
    }

    #[test]
    fn sparse_outcome_produces_warnings() {
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
                ResolveWarning::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: policy.minimum_valid_responses,
                },
            ]
        );
    }

    #[test]
    fn invalid_and_error_responses_produce_warnings() {
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

        assert_eq!(
            classify(&report),
            vec![
                ResolveWarning::InvalidResponses {
                    invalid_values: 1,
                    invalid_responses: 1,
                    invalid_signed_packets: 0,
                },
                ResolveWarning::KrpcErrors { krpc_errors: 1 },
            ]
        );
    }

    #[test]
    fn invalid_signed_packets_produce_warnings() {
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

        assert_eq!(
            classify(&report),
            vec![ResolveWarning::InvalidResponses {
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

        assert_eq!(report.valid_responses(), 0);
        let policy = ReportPolicy::mainnet();
        assert_eq!(
            classify(&report),
            vec![
                ResolveWarning::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: policy.minimum_valid_responses,
                },
                ResolveWarning::InvalidResponses {
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
        let policy = ReportPolicy::mainnet();
        assert_eq!(
            classify(&report),
            vec![
                ResolveWarning::TooFewNodesResponded {
                    responded: 1,
                    minimum: policy.minimum_responded_nodes,
                },
                ResolveWarning::TooFewValidResponses {
                    valid_responses: 0,
                    minimum: policy.minimum_valid_responses,
                },
                ResolveWarning::InvalidResponses {
                    invalid_values: 0,
                    invalid_responses: 0,
                    invalid_signed_packets: 2,
                },
            ]
        );
    }

    #[test]
    fn testnet_policy_relaxes_topology_thresholds() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 2,
                values: 0,
                no_values: 1,
                no_more_recent: 0,
                invalid_values: 0,
                invalid_responses: 0,
                krpc_errors: 0,
            },
            0,
        );

        assert!(ReportPolicy::testnet()
            .classify_resolve_report(&report)
            .is_empty());
    }

    #[test]
    fn custom_policy_controls_warning_thresholds() {
        let report = ResolveReport::with_invalid_signed_packets(
            mainline::GetMutableOutcome {
                queried: 2,
                values: 0,
                no_values: 1,
                no_more_recent: 0,
                invalid_values: 1,
                invalid_responses: 0,
                krpc_errors: 1,
            },
            0,
        );

        let warnings = ReportPolicy {
            minimum_publish_stored_nodes: 1,
            minimum_queried_nodes: 2,
            minimum_responded_nodes: 1,
            minimum_valid_responses: 1,
            report_invalid_responses: false,
            report_krpc_errors: false,
        }
        .classify_resolve_report(&report);

        assert!(warnings.is_empty());
    }
}
