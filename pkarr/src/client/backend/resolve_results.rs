use std::collections::HashMap;

use crate::client::ResolveError;
use crate::SignedPacket;

#[derive(Default)]
pub(super) struct ResolveResults {
    highest_invalid_signed_packet_seq: Option<i64>,
    most_recent: Option<SignedPacket>,
    errors: HashMap<ResolveError, usize>,
    empty_responses: usize,
}

impl ResolveResults {
    pub(super) fn record(&mut self, result: Result<SignedPacket, ResolveError>) -> bool {
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

    pub(super) fn finish(self) -> Result<SignedPacket, ResolveError> {
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
