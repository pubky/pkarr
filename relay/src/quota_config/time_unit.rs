use std::time::Duration;

/// The time unit of the quota.
///
/// Examples:
/// - "s" -> second
/// - "m" -> minute
/// - "h" -> hour
/// - "d" -> day
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Second
    Second,
    /// Minute
    Minute,
    /// Hour
    Hour,
    /// Day
    Day,
}

impl std::fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unit = match self {
            TimeUnit::Second => "s",
            TimeUnit::Minute => "m",
            TimeUnit::Hour => "h",
            TimeUnit::Day => "d",
        };
        write!(f, "{unit}")
    }
}

impl std::str::FromStr for TimeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "s" => Ok(TimeUnit::Second),
            "m" => Ok(TimeUnit::Minute),
            "h" => Ok(TimeUnit::Hour),
            "d" => Ok(TimeUnit::Day),
            _ => Err(format!("Invalid time unit: {s}")),
        }
    }
}

impl From<TimeUnit> for Duration {
    fn from(time_unit: TimeUnit) -> Self {
        match time_unit {
            TimeUnit::Second => Duration::from_secs(1),
            TimeUnit::Minute => Duration::from_secs(60),
            TimeUnit::Hour => Duration::from_secs(60 * 60),
            TimeUnit::Day => Duration::from_secs(24 * 60 * 60),
        }
    }
}
