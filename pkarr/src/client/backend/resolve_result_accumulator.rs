use std::collections::HashMap;

use crate::client::ResolveError;
use crate::SignedPacket;

/// Accumulates backend resolve responses and selects the result returned to the client.
#[derive(Default)]
pub(super) struct ResolveResultAccumulator {
    highest_invalid_signed_packet_seq: Option<i64>,
    most_recent: Option<SignedPacket>,
    errors: HashMap<ResolveError, usize>,
    empty_responses: usize,
}

impl ResolveResultAccumulator {
    /// Tracks one backend response and returns whether it contained a usable packet.
    ///
    /// Successful packets are kept by newest timestamp. `NotFound` responses are
    /// counted separately from other errors so that an empty network result can
    /// be returned when no backend produced a packet. Invalid signed-packet
    /// sequence numbers are tracked separately because a newer invalid DHT
    /// mutable item must outrank an older valid packet.
    pub(super) fn record_result(&mut self, result: Result<SignedPacket, ResolveError>) -> bool {
        match result {
            Ok(packet) => {
                self.most_recent = Some(most_recent_packet(self.most_recent.take(), packet));
                true
            }
            Err(ResolveError::NotFound) => {
                self.empty_responses += 1;
                false
            }
            Err(ResolveError::InvalidSignedPacket { seq }) => {
                self.highest_invalid_signed_packet_seq = Some(
                    self.highest_invalid_signed_packet_seq
                        .unwrap_or(seq)
                        .max(seq),
                );
                false
            }
            Err(error) => {
                *self.errors.entry(error).or_default() += 1;
                false
            }
        }
    }

    /// Converts all recorded backend responses into the resolve result returned to callers.
    pub(super) fn into_result(self) -> Result<SignedPacket, ResolveError> {
        match (self.highest_invalid_signed_packet_seq, self.most_recent) {
            (None, Some(packet)) => Ok(packet),
            (Some(seq), Some(packet)) if packet.timestamp().as_u64() as i64 >= seq => Ok(packet),
            (Some(seq), _) => Err(ResolveError::InvalidSignedPacket { seq }),
            (None, None) => {
                if self.empty_responses > 0 {
                    Err(ResolveError::NotFound)
                } else {
                    Err(most_common_error(self.errors, ResolveError::UnexpectedResponses).0)
                }
            }
        }
    }
}

fn most_common_error<T>(errors: HashMap<T, usize>, fallback: T) -> (T, usize)
where
    T: Eq + std::hash::Hash,
{
    errors
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .unwrap_or((fallback, 0))
}

fn most_recent_packet(most_recent: Option<SignedPacket>, packet: SignedPacket) -> SignedPacket {
    match most_recent {
        Some(most_recent) if most_recent.more_recent_than(&packet) => most_recent,
        _ => packet,
    }
}
