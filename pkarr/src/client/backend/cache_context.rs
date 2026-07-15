use ntimestamp::Timestamp;

use crate::SignedPacket;

const HTTP_DATE_SECOND_MICROS: u64 = 1_000_000;

/// Cached packet floor and local TTL bounds used during cache-first resolution.
#[derive(Clone, Copy, Debug)]
pub(in crate::client) struct CacheContext<'a> {
    cached: Option<&'a SignedPacket>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl<'a> CacheContext<'a> {
    /// Creates cache-first state from the cached packet and client TTL limits.
    pub(in crate::client) const fn new(
        cached: Option<&'a SignedPacket>,
        minimum_ttl: u32,
        maximum_ttl: u32,
    ) -> Self {
        Self {
            cached,
            minimum_ttl,
            maximum_ttl,
        }
    }

    /// Returns whether `packet` is ordered below the cached packet floor.
    pub(super) fn packet_is_below_floor(self, packet: &SignedPacket) -> bool {
        self.cached
            .is_some_and(|cached| cached.more_recent_than(packet))
    }

    /// Returns whether the cached packet covers an invalid mutable-item sequence.
    pub(super) fn invalid_seq_is_covered(self, seq: i64) -> bool {
        seq >= 0
            && self
                .cached
                .is_some_and(|cached| cached.timestamp().as_u64() >= seq as u64)
    }

    /// Returns a DHT request bound just below the cached packet so equal-sequence
    /// values remain eligible for packet-value tie-breaking.
    ///
    /// Returns [`None`] when there is no cached packet or its timestamp cannot
    /// be lowered.
    #[cfg(dht)]
    pub(super) fn dht_request_lower_bound(self) -> Option<Timestamp> {
        self.cached?
            .timestamp()
            .as_u64()
            .checked_sub(1)
            .map(Timestamp::from)
    }

    /// Returns a relay request bound in the HTTP second before the cached packet.
    ///
    /// HTTP dates have one-second precision, so subtracting only one microsecond
    /// could serialize to the same `If-Modified-Since` value as the cached packet.
    pub(super) fn relay_request_lower_bound(self) -> Option<Timestamp> {
        self.cached?
            .timestamp()
            .as_u64()
            .checked_sub(HTTP_DATE_SECOND_MICROS)
            .map(Timestamp::from)
    }

    /// Returns whether `packet` is fresh enough to return from cache-first
    /// resolution.
    ///
    /// A maximum TTL of zero permits network results while local cache hits are
    /// bypassed by the client.
    pub(in crate::client) fn accepts_network_packet(self, packet: &SignedPacket) -> bool {
        self.maximum_ttl == 0 || !packet.is_expired(self.minimum_ttl, self.maximum_ttl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Keypair;

    #[test]
    fn invalid_sequence_must_be_non_negative_and_above_the_cached_sequence() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(10))
            .sign(&Keypair::random())
            .unwrap();
        let context = CacheContext::new(Some(&cached), 0, 0);

        assert!(!context.invalid_seq_is_covered(-1));
        assert!(context.invalid_seq_is_covered(10));
        assert!(!context.invalid_seq_is_covered(11));
    }

    #[cfg(dht)]
    #[test]
    fn zero_timestamp_has_no_dht_request_lower_bound() {
        let cached = SignedPacket::builder()
            .timestamp(Timestamp::from(0))
            .sign(&Keypair::random())
            .unwrap();

        assert_eq!(
            CacheContext::new(Some(&cached), 0, 0).dht_request_lower_bound(),
            None
        );
    }

    #[test]
    fn relay_request_lower_bound_uses_previous_http_second() {
        let timestamp =
            Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap() + 500_000;
        let cached = SignedPacket::builder()
            .timestamp(timestamp)
            .sign(&Keypair::random())
            .unwrap();
        let context = CacheContext::new(Some(&cached), 0, 0);

        #[cfg(dht)]
        assert_eq!(context.dht_request_lower_bound(), Some(timestamp - 1));
        assert_eq!(
            context
                .relay_request_lower_bound()
                .unwrap()
                .format_http_date(),
            "Sun, 06 Nov 1994 08:49:36 GMT"
        );
    }

    #[test]
    fn packet_floor_uses_same_sequence_packet_ordering() {
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
        let (floor, below_floor) = if first.more_recent_than(&second) {
            (first, second)
        } else {
            (second, first)
        };

        assert!(CacheContext::new(Some(&floor), 0, 0).packet_is_below_floor(&below_floor));
        assert!(!CacheContext::new(Some(&below_floor), 0, 0).packet_is_below_floor(&floor));
    }

    #[test]
    fn zero_maximum_ttl_accepts_expired_network_packet() {
        let mut packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&Keypair::random())
            .unwrap();
        packet.set_last_seen(&(Timestamp::now() - 60 * 1_000_000_u64));

        assert!(CacheContext::new(None, 0, 0).accepts_network_packet(&packet));
        assert!(!CacheContext::new(None, 0, 30).accepts_network_packet(&packet));
    }
}
