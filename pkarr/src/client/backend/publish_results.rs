use std::collections::HashMap;

use crate::client::{ConcurrencyError, PublishError};

#[derive(Default)]
pub(super) struct PublishResults {
    successes: usize,
    max_stored_on: u32,
    errors: HashMap<PublishError, usize>,
}

impl PublishResults {
    pub(super) fn record(&mut self, result: Result<u32, PublishError>) {
        match result {
            Ok(stored_on) => {
                self.successes += 1;
                self.max_stored_on = self.max_stored_on.max(stored_on);
            }
            Err(error) => *self.errors.entry(error).or_default() += 1,
        }
    }

    pub(super) fn finish(self) -> Result<u32, PublishError> {
        if let Some(error) = self.blocking_concurrency_error() {
            return Err(error);
        }

        let (error, error_count) =
            most_common_error(self.errors, PublishError::UnexpectedResponses);

        match error {
            // A relay CAS failure can be self-induced: one relay gateway may
            // publish to the shared DHT before another gateway handles the same
            // CAS request. Accept that split when successes cover the failures.
            PublishError::Concurrency(_) if self.successes > 0 && self.successes >= error_count => {
                Ok(self.max_stored_on)
            }
            PublishError::Concurrency(_) => Err(error),
            _ if self.successes > 0 => Ok(self.max_stored_on),
            _ => Err(error),
        }
    }

    fn blocking_concurrency_error(&self) -> Option<PublishError> {
        [
            ConcurrencyError::ConflictRisk,
            ConcurrencyError::NotMostRecent,
        ]
        .into_iter()
        .map(PublishError::Concurrency)
        .find(|error| {
            self.errors
                .get(error)
                .is_some_and(|count| *count >= self.successes)
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::QueryError;

    #[test]
    fn finish_checks_all_high_priority_concurrency_errors() {
        let mut results = PublishResults::default();
        for result in [
            Ok(3),
            Ok(5),
            Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent)),
            Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent)),
            Err(PublishError::Concurrency(ConcurrencyError::ConflictRisk)),
            Err(PublishError::Query(QueryError::Timeout)),
            Err(PublishError::Query(QueryError::Timeout)),
            Err(PublishError::Query(QueryError::Timeout)),
        ] {
            results.record(result);
        }

        assert_eq!(
            results.finish(),
            Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent))
        );
    }

    #[test]
    fn finish_accepts_cas_failure_tie_after_success() {
        let mut results = PublishResults::default();
        results.record(Ok(7));
        results.record(Err(PublishError::Concurrency(ConcurrencyError::CasFailed)));

        assert_eq!(results.finish(), Ok(7));
    }

    #[test]
    fn finish_returns_maximum_stored_on_count() {
        let mut results = PublishResults::default();
        results.record(Ok(3));
        results.record(Ok(11));
        results.record(Ok(7));

        assert_eq!(results.finish(), Ok(11));
    }
}
