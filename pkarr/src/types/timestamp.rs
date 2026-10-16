use std::{
    fmt,
    ops::{Add, Sub, SubAssign},
    sync::Mutex,
    time::{Duration, SystemTime},
};

static LAST_TIMESTAMP: Mutex<u64> = Mutex::new(0);

// Last microsecond of year 9999; httpdate panics on later dates.
const MAX_HTTP_TIMESTAMP: Timestamp = Timestamp(253_402_300_799_999_999);

/// Microseconds since the Unix epoch, encoded as eight big-endian bytes.
///
/// [`Self::now`] produces strictly increasing values within this process,
/// except at [`u64::MAX`]. Values constructed explicitly are used as supplied.
/// Addition and subtraction use microseconds and saturate at zero and
/// [`u64::MAX`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Return the current time, advancing by at least one microsecond since
    /// the previous call even if the wall clock repeats or moves backward.
    /// Saturates at [`u64::MAX`].
    pub fn now() -> Self {
        #[cfg(not(wasm_browser))]
        let micros = Self::from(SystemTime::now()).0;
        #[cfg(wasm_browser)]
        let micros = (js_sys::Date::now() * 1_000.0) as u64;

        let mut last = LAST_TIMESTAMP
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *last = next_timestamp(micros, *last);
        Self(*last)
    }

    /// Return the number of microseconds since the Unix epoch.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Return the timestamp as eight big-endian bytes.
    pub const fn to_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    /// Parse an HTTP date, which has whole-second precision.
    pub fn parse_http_date(date: &str) -> Result<Self, httpdate::Error> {
        httpdate::parse_http_date(date).map(Self::from)
    }

    /// Format an HTTP date, discarding subsecond precision.
    ///
    /// Returns `None` for dates beyond year 9999, which HTTP dates cannot
    /// represent.
    pub fn format_http_date(self) -> Option<String> {
        if self > MAX_HTTP_TIMESTAMP {
            return None;
        }
        Some(httpdate::fmt_http_date(self.into()))
    }

    /// Format an HTTP date, clamping dates beyond year 9999 to
    /// `Fri, 31 Dec 9999 23:59:59 GMT` and discarding subsecond precision.
    pub fn format_http_date_saturating(self) -> String {
        httpdate::fmt_http_date(self.min(MAX_HTTP_TIMESTAMP).into())
    }
}

fn next_timestamp(micros: u64, last: u64) -> u64 {
    micros.max(last.saturating_add(1))
}

impl From<u64> for Timestamp {
    fn from(micros: u64) -> Self {
        Self(micros)
    }
}

impl From<[u8; 8]> for Timestamp {
    fn from(bytes: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(bytes))
    }
}

impl From<SystemTime> for Timestamp {
    /// Convert to microseconds, clamping times before the epoch to zero and
    /// times beyond the supported range to [`u64::MAX`].
    fn from(time: SystemTime) -> Self {
        let micros = time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        Self(u64::try_from(micros).unwrap_or(u64::MAX))
    }
}

impl From<Timestamp> for SystemTime {
    fn from(timestamp: Timestamp) -> Self {
        Self::UNIX_EPOCH + Duration::from_micros(timestamp.0)
    }
}

impl From<httpdate::HttpDate> for Timestamp {
    fn from(date: httpdate::HttpDate) -> Self {
        Self::from(SystemTime::from(date))
    }
}

impl Add<u64> for Timestamp {
    type Output = Self;

    fn add(self, micros: u64) -> Self {
        Self(self.0.saturating_add(micros))
    }
}

impl Sub<u64> for Timestamp {
    type Output = Self;

    fn sub(self, micros: u64) -> Self {
        Self(self.0.saturating_sub(micros))
    }
}

impl SubAssign<u64> for Timestamp {
    fn sub_assign(&mut self, micros: u64) {
        *self = *self - micros;
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Preserve the sortable string representation used by ntimestamp.
        formatter.write_str(&base32::encode(
            base32::Alphabet::Crockford,
            &self.to_bytes(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(wasm_browser)]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn clock_repetition_and_rollback_still_advance() {
        assert_eq!(next_timestamp(100, 90), 100);
        assert_eq!(next_timestamp(100, 100), 101);
        assert_eq!(next_timestamp(90, 100), 101);
        assert_eq!(next_timestamp(0, u64::MAX), u64::MAX);
        let first = Timestamp::now();
        assert!(Timestamp::now() > first);
    }

    #[test]
    fn bytes_preserve_big_endian_microseconds() {
        for value in [0, 1, 1_234_567_890_123_456, u64::MAX] {
            let timestamp = Timestamp::from(value);
            let bytes = value.to_be_bytes();
            assert_eq!(timestamp.to_bytes(), bytes);
            assert_eq!(Timestamp::from(bytes), timestamp);
        }
    }

    #[test]
    fn system_time_preserves_microseconds_and_clamps_before_epoch() {
        let time = SystemTime::UNIX_EPOCH + Duration::from_nanos(1_234_567_890);
        let timestamp = Timestamp::from(time);
        assert_eq!(timestamp.as_u64(), 1_234_567);
        assert_eq!(
            SystemTime::from(timestamp),
            time - Duration::from_nanos(890)
        );
        // Some platforms, including browser WASM, cannot represent pre-epoch times.
        if let Some(before_epoch) = SystemTime::UNIX_EPOCH.checked_sub(Duration::from_secs(1)) {
            assert_eq!(Timestamp::from(before_epoch), Timestamp::from(0));
        }
    }

    #[test]
    fn http_dates_discard_only_subsecond_precision() {
        let date = "Sun, 06 Nov 1994 08:49:37 GMT";
        let timestamp = Timestamp::parse_http_date(date).unwrap();
        assert_eq!(timestamp.as_u64(), 784_111_777_000_000);
        assert_eq!(
            (timestamp + 999_999).format_http_date().as_deref(),
            Some(date)
        );
        let http_date = httpdate::HttpDate::from(SystemTime::from(timestamp));
        assert_eq!(Timestamp::from(http_date), timestamp);
        assert!(Timestamp::parse_http_date("invalid").is_err());
    }

    #[test]
    fn http_dates_handle_the_full_timestamp_range() {
        let last_second = Timestamp::parse_http_date("Fri, 31 Dec 9999 23:59:59 GMT").unwrap();
        assert_eq!(
            (last_second + 999_999).format_http_date().as_deref(),
            Some("Fri, 31 Dec 9999 23:59:59 GMT")
        );
        assert_eq!((last_second + 1_000_000).format_http_date(), None);
        assert_eq!(Timestamp::from(u64::MAX).format_http_date(), None);
        assert_eq!(
            Timestamp::from(0).format_http_date().as_deref(),
            Some("Thu, 01 Jan 1970 00:00:00 GMT")
        );
    }

    #[test]
    fn saturating_http_dates_clamp_only_beyond_the_supported_range() {
        let max_date = "Fri, 31 Dec 9999 23:59:59 GMT";
        let last_second = Timestamp::parse_http_date(max_date).unwrap();
        for timestamp in [
            last_second,
            last_second + 999_999,
            last_second + 1_000_000,
            Timestamp::from(u64::MAX),
        ] {
            assert_eq!(timestamp.format_http_date_saturating(), max_date);
        }
        let date = "Sun, 06 Nov 1994 08:49:37 GMT";
        let timestamp = Timestamp::parse_http_date(date).unwrap() + 999_999;
        assert_eq!(timestamp.format_http_date_saturating(), date);
        assert_eq!(
            Timestamp::from(0).format_http_date_saturating(),
            "Thu, 01 Jan 1970 00:00:00 GMT"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn concurrent_calls_produce_unique_increasing_timestamps() {
        let worker_count = 4;
        let barrier = std::sync::Barrier::new(worker_count);
        let mut timestamps = Vec::new();

        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for _ in 0..worker_count {
                workers.push(scope.spawn(|| {
                    barrier.wait();
                    let timestamps: Vec<_> = (0..1_000).map(|_| Timestamp::now()).collect();
                    assert!(timestamps.windows(2).all(|pair| pair[0] < pair[1]));
                    timestamps
                }));
            }
            for worker in workers {
                timestamps.extend(worker.join().unwrap());
            }
        });

        timestamps.sort_unstable();
        assert!(timestamps.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn arithmetic_saturates_at_the_timestamp_bounds() {
        assert_eq!(Timestamp::from(0) - 1, Timestamp::from(0));
        assert_eq!(Timestamp::from(u64::MAX) + 1, Timestamp::from(u64::MAX));
        assert_eq!((Timestamp::from(42) + 10) - 5, Timestamp::from(47));
    }
}
