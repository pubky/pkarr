use std::cmp::Ordering;
use std::collections::HashMap;

use crate::client::ResolveError;
use crate::SignedPacket;

use super::CacheContext;

/// Accumulates backend resolve responses, applies optional cache-first
/// constraints, and selects the backend result.
///
/// Expired packets above the cached floor remain fallback cache candidates,
/// while fresh packets are tracked separately for cache-first completion.
#[derive(Default)]
pub(super) struct ResolveResultAccumulator<'a> {
    cache_context: Option<CacheContext<'a>>,
    highest_invalid_signed_packet_seq: Option<i64>,
    most_recent: Option<SignedPacket>,
    fresh_candidate: Option<SignedPacket>,
    errors: HashMap<ResolveError, usize>,
    empty_responses: usize,
}

impl<'a> ResolveResultAccumulator<'a> {
    /// Creates an accumulator with optional cache-first constraints.
    pub(super) fn new(cache_context: Option<CacheContext<'a>>) -> Self {
        Self {
            cache_context,
            ..Self::default()
        }
    }

    /// Tracks one backend response and returns whether that response may
    /// complete resolution under the configured cache-first constraints.
    ///
    /// Packets above the cached floor are kept by [`SignedPacket`] freshness
    /// ordering, including expired packets retained as cache-update fallbacks.
    /// Fresh cache-first responses are tracked separately so a newer expired
    /// packet cannot hide an acceptable fresh packet. When packets are
    /// identical, the observation with the newest [`SignedPacket::last_seen`]
    /// is retained. `NotFound` and floor-covered responses are counted as empty,
    /// while other errors are counted by kind. Invalid signed-packet sequence
    /// numbers above the floor are tracked separately because a newer invalid
    /// DHT mutable item must outrank an older valid packet.
    pub(super) fn record_result(&mut self, result: Result<SignedPacket, ResolveError>) -> bool {
        match result {
            Ok(packet)
                if self
                    .cache_context
                    .is_some_and(|context| context.packet_is_below_floor(&packet)) =>
            {
                self.empty_responses += 1;
                false
            }
            Ok(packet) => {
                let is_acceptable = self
                    .cache_context
                    .is_none_or(|context| context.accepts_network_packet(&packet));

                if is_acceptable && self.cache_context.is_some() {
                    self.fresh_candidate = Some(most_recent_packet(
                        self.fresh_candidate.take(),
                        packet.clone(),
                    ));
                }

                self.most_recent = Some(most_recent_packet(self.most_recent.take(), packet));
                is_acceptable
            }
            Err(ResolveError::NotFound) => {
                self.empty_responses += 1;
                false
            }
            Err(ResolveError::InvalidSignedPacket { seq }) => {
                if self
                    .cache_context
                    .is_some_and(|context| context.invalid_seq_is_covered(seq))
                {
                    self.empty_responses += 1;
                    return false;
                }
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

    /// Converts all recorded responses into the selected packet or resolve error.
    ///
    /// A fresh cache-first candidate is preferred over expired packets retained
    /// as cache-update fallbacks. A newer invalid mutable-item sequence outranks
    /// the selected valid packet. With cache-first constraints, an expired
    /// packet may be returned when no fresh candidate exists so the client can
    /// update its cache before returning [`ResolveError::NotFound`] to its caller.
    /// If every usable response was empty or covered by the floor, returns
    /// [`ResolveError::NotFound`].
    pub(super) fn into_result(self) -> Result<SignedPacket, ResolveError> {
        let packet = self.fresh_candidate.or(self.most_recent);

        match (self.highest_invalid_signed_packet_seq, packet) {
            (None, Some(packet)) => Ok(packet),
            (Some(seq), Some(packet)) if packet.timestamp().as_u64() as i64 >= seq => Ok(packet),
            (Some(seq), _) => Err(ResolveError::InvalidSignedPacket { seq }),
            (None, None) => {
                if self.empty_responses > 0 {
                    Err(ResolveError::NotFound)
                } else {
                    Err(most_common_error(self.errors))
                }
            }
        }
    }
}

fn most_common_error(error_counts: HashMap<ResolveError, usize>) -> ResolveError {
    error_counts
        .into_iter()
        .max_by(|(left_error, left_count), (right_error, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| compare_errors(left_error, right_error))
        })
        .map_or(ResolveError::UnexpectedResponses, |(error, _)| error)
}

fn compare_errors(left: &ResolveError, right: &ResolveError) -> Ordering {
    let priority = |error: &ResolveError| match error {
        ResolveError::UnexpectedResponses => 0,
        ResolveError::NoResponses => 1,
        ResolveError::NoDhtNodesQueried => 2,
        ResolveError::NoUsableResponses => 3,
        ResolveError::NotFound => 4,
        ResolveError::InvalidSignedPacket { .. } => 5,
    };

    priority(left)
        .cmp(&priority(right))
        .then_with(|| match (left, right) {
            (
                ResolveError::InvalidSignedPacket { seq: left_seq },
                ResolveError::InvalidSignedPacket { seq: right_seq },
            ) => left_seq.cmp(right_seq),
            _ => Ordering::Equal,
        })
}

fn most_recent_packet(most_recent: Option<SignedPacket>, packet: SignedPacket) -> SignedPacket {
    match most_recent {
        Some(most_recent)
            if most_recent.more_recent_than(&packet)
                || (most_recent.is_same_as(&packet)
                    && most_recent.last_seen() > packet.last_seen()) =>
        {
            most_recent
        }
        _ => packet,
    }
}

#[cfg(test)]
mod tests {
    use ntimestamp::Timestamp;

    use super::*;
    use crate::Keypair;

    #[test]
    fn record_result_waits_for_packet_above_cache_floor() {
        let keypair = Keypair::random();
        let first = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "first".try_into().unwrap(), 30)
            .timestamp(Timestamp::from(10))
            .sign(&keypair)
            .unwrap();
        let second = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "second".try_into().unwrap(), 30)
            .timestamp(Timestamp::from(10))
            .sign(&keypair)
            .unwrap();
        let (cached, below_floor) = if first.more_recent_than(&second) {
            (first, second)
        } else {
            (second, first)
        };
        let above_floor = SignedPacket::builder()
            .timestamp(Timestamp::from(11))
            .sign(&keypair)
            .unwrap();

        let mut accumulator =
            ResolveResultAccumulator::new(Some(CacheContext::new(Some(&cached), 0, 0)));
        assert!(!accumulator.record_result(Ok(below_floor)));
        assert!(accumulator.record_result(Ok(above_floor.clone())));

        assert_eq!(accumulator.into_result(), Ok(above_floor));
    }

    #[test]
    fn into_result_treats_equal_invalid_seq_as_covered_by_cache() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();
        let mut accumulator =
            ResolveResultAccumulator::new(Some(CacheContext::new(Some(&cached), 0, 0)));

        accumulator.record_result(Err(ResolveError::InvalidSignedPacket { seq: 10 }));

        assert_eq!(accumulator.into_result(), Err(ResolveError::NotFound));
    }

    #[test]
    fn into_result_preserves_freshest_last_seen_for_identical_packets() {
        let mut stale = SignedPacket::builder().sign(&Keypair::random()).unwrap();
        stale.set_last_seen(&Timestamp::from(10));
        let mut fresh = stale.clone();
        fresh.set_last_seen(&Timestamp::from(20));

        let mut accumulator = ResolveResultAccumulator::default();
        accumulator.record_result(Ok(fresh));
        accumulator.record_result(Ok(stale));
        let resolved = accumulator.into_result().unwrap();

        assert_eq!(resolved.last_seen(), &Timestamp::from(20));
    }

    #[test]
    fn fresh_candidate_is_not_hidden_by_newer_expired_packet() {
        let keypair = Keypair::random();
        let mut cached = SignedPacket::builder()
            .timestamp(Timestamp::from(9))
            .sign(&keypair)
            .unwrap();
        cached.set_last_seen(&(Timestamp::now() - 60 * 1_000_000_u64));
        let mut expired = SignedPacket::builder()
            .timestamp(Timestamp::from(11))
            .sign(&keypair)
            .unwrap();
        expired.set_last_seen(&(Timestamp::now() - 60 * 1_000_000_u64));
        let mut fresh = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "fresh".try_into().unwrap(), 30)
            .timestamp(Timestamp::from(10))
            .sign(&keypair)
            .unwrap();
        fresh.set_last_seen(&Timestamp::now());

        let mut accumulator =
            ResolveResultAccumulator::new(Some(CacheContext::new(Some(&cached), 0, 30)));
        assert!(!accumulator.record_result(Ok(expired)));
        assert!(accumulator.record_result(Ok(fresh.clone())));

        assert_eq!(accumulator.into_result(), Ok(fresh));
    }

    #[test]
    fn into_result_prefers_more_specific_error_when_counts_tie() {
        let mut accumulator = ResolveResultAccumulator::default();
        accumulator.record_result(Err(ResolveError::NoResponses));
        accumulator.record_result(Err(ResolveError::NoUsableResponses));

        assert_eq!(
            accumulator.into_result(),
            Err(ResolveError::NoUsableResponses)
        );
    }
}
