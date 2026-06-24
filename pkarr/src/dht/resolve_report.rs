/// Diagnostics for a DHT resolve query.
///
/// This wraps Mainline's raw mutable GET outcome so pkarr can expose stable
/// diagnostics without exposing the Mainline type directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveReport(pub(super) mainline::GetMutableOutcome);

impl ResolveReport {
    /// Build a report from a raw Mainline mutable GET outcome.
    pub(crate) fn new(outcome: mainline::GetMutableOutcome) -> Self {
        Self(outcome)
    }

    /// Number of unique DHT nodes queried.
    pub fn queried(&self) -> u32 {
        self.0.queried
    }

    /// Number of nodes that returned a GET response before timing out.
    ///
    /// This includes malformed responses and KRPC error responses; use
    /// `ReportPolicy` to classify those counters into warnings.
    pub fn responded(&self) -> u32 {
        self.0.responded()
    }

    /// Number of usable responses.
    pub fn usable_responses(&self) -> u32 {
        self.0.valid_responses()
    }
}
