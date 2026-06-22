use std::{fmt, future::Future, pin::Pin};

use super::resolve_report::ResolveReport;
use crate::SignedPacket;

type ResolveCompletion = Pin<Box<dyn Future<Output = ResolveOutcome> + Send + 'static>>;

/// Successful DHT resolve response returned after the first valid signed packet is found.
pub struct ResolveResponse {
    first: SignedPacket,
    completion: ResolveCompletion,
}

impl ResolveResponse {
    pub(crate) fn new(
        first: SignedPacket,
        completion: impl Future<Output = ResolveOutcome> + Send + 'static,
    ) -> Self {
        Self {
            first,
            completion: Box::pin(completion),
        }
    }

    /// First valid signed packet returned by the DHT query.
    pub fn first(&self) -> &SignedPacket {
        &self.first
    }

    /// Finish the DHT query and return the most recent value and diagnostics.
    pub async fn complete(self) -> ResolveOutcome {
        self.completion.await
    }
}

impl fmt::Debug for ResolveResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolveResponse")
            .field("first", &self.first)
            .finish_non_exhaustive()
    }
}

/// Completed DHT resolve result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveOutcome {
    /// Most recent mutable item returned by the DHT query.
    pub most_recent: ResolveValue,
    /// Diagnostics collected from the DHT query.
    pub report: ResolveReport,
}

/// Most recent mutable item returned by a completed DHT resolve query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveValue {
    /// The most recent mutable item contains a valid signed packet.
    ValidSignedPacket {
        /// Valid signed packet contained by the mutable item.
        packet: SignedPacket,
    },
    /// The most recent mutable item is newer than all valid signed packets, but
    /// does not contain a valid signed packet.
    ///
    /// Older valid signed packets must not be treated as current when this
    /// variant is returned; the newer invalid mutable item is the current DHT
    /// state for the key.
    InvalidSignedPacket {
        /// Mutable item sequence number.
        seq: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_packet() -> SignedPacket {
        crate::SignedPacket::builder()
            .address(
                "_derp_region.iroh.".try_into().unwrap(),
                "1.1.1.1".parse().unwrap(),
                30,
            )
            .sign(&crate::Keypair::random())
            .unwrap()
    }

    fn healthy_report() -> ResolveReport {
        ResolveReport::new(mainline::GetMutableOutcome {
            queried: 20,
            values: 1,
            no_values: 2,
            no_more_recent: 0,
            invalid_values: 0,
            invalid_responses: 0,
            krpc_errors: 0,
        })
    }

    #[test]
    fn resolve_response_exposes_first_packet() {
        let first = signed_packet();
        let response = ResolveResponse::new(first.clone(), async move {
            ResolveOutcome {
                most_recent: ResolveValue::ValidSignedPacket {
                    packet: signed_packet(),
                },
                report: healthy_report(),
            }
        });

        assert_eq!(response.first(), &first);
    }

    #[test]
    fn resolve_response_completion_returns_resolve_outcome() {
        let first = signed_packet();
        let most_recent = signed_packet();
        let report = healthy_report();
        let expected_packet = most_recent.clone();
        let expected_report = report.clone();
        let response = ResolveResponse::new(first, async move {
            ResolveOutcome {
                most_recent: ResolveValue::ValidSignedPacket {
                    packet: most_recent,
                },
                report,
            }
        });

        let resolved = futures_lite::future::block_on(response.complete());

        assert_eq!(
            resolved.most_recent,
            ResolveValue::ValidSignedPacket {
                packet: expected_packet
            }
        );
        assert_eq!(resolved.report, expected_report);
    }
}
