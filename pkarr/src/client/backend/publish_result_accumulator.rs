use std::collections::HashMap;

use crate::client::PublishError;
use crate::StoredNodeCount;

/// Accumulates backend publish responses and selects the result returned to the client.
#[derive(Default)]
pub(super) struct PublishResultAccumulator {
    successes: usize,
    max_stored_on: StoredNodeCount,
    errors: HashMap<PublishError, usize>,
}

impl PublishResultAccumulator {
    /// Tracks one backend response.
    ///
    /// Successful publishes contribute to the maximum stored-node count returned
    /// to callers. Errors are counted by kind so the final result can prefer
    /// blocking concurrency errors or otherwise return the most common error.
    pub(super) fn record_result(&mut self, result: Result<StoredNodeCount, PublishError>) {
        match result {
            Ok(stored_on) => {
                self.successes += 1;
                self.max_stored_on = self.max_stored_on.max(stored_on);
            }
            Err(error) => *self.errors.entry(error).or_default() += 1,
        }
    }

    /// Converts all recorded backend responses into the publish result returned to callers.
    pub(super) fn into_result(self) -> Result<StoredNodeCount, PublishError> {
        if let Some(error) = self.blocking_concurrency_error() {
            return Err(error);
        }

        let (error, error_count) =
            most_common_error(self.errors, PublishError::UnexpectedResponses);

        match error {
            PublishError::NotMostRecent if self.successes > 0 && self.successes >= error_count => {
                Ok(self.max_stored_on)
            }
            PublishError::NotMostRecent => Err(error),
            _ if self.successes > 0 => Ok(self.max_stored_on),
            _ => Err(error),
        }
    }

    fn blocking_concurrency_error(&self) -> Option<PublishError> {
        let error = PublishError::NotMostRecent;
        self.errors
            .get(&error)
            .is_some_and(|count| *count >= self.successes)
            .then_some(error)
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

    #[test]
    fn into_result_prefers_blocking_concurrency_errors() {
        let mut accumulator = PublishResultAccumulator::default();
        for result in [
            Ok(3),
            Ok(5),
            Err(PublishError::NotMostRecent),
            Err(PublishError::NotMostRecent),
            Err(PublishError::NoResponses),
            Err(PublishError::NoResponses),
            Err(PublishError::NoResponses),
        ] {
            accumulator.record_result(result);
        }

        assert_eq!(accumulator.into_result(), Err(PublishError::NotMostRecent));
    }

    #[test]
    fn into_result_returns_maximum_stored_on_count() {
        let mut accumulator = PublishResultAccumulator::default();
        accumulator.record_result(Ok(3));
        accumulator.record_result(Ok(11));
        accumulator.record_result(Ok(7));

        assert_eq!(accumulator.into_result(), Ok(11));
    }
}
