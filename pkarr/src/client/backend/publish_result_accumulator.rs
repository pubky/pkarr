use std::cmp::Ordering;
use std::collections::HashMap;

use crate::client::PublishError;
use crate::StoredNodeCount;

/// Accumulates backend publish responses and selects the result returned to the client.
#[derive(Default)]
pub(super) struct PublishResultAccumulator {
    success_count: usize,
    max_stored_nodes: StoredNodeCount,
    error_counts: HashMap<PublishError, usize>,
}

impl PublishResultAccumulator {
    /// Tracks one backend response.
    ///
    /// Successful publishes contribute to the maximum stored-node count returned
    /// to callers. Errors are counted by exact value so the final result can
    /// prefer blocking concurrency errors or otherwise return the most common
    /// error.
    pub(super) fn record_result(&mut self, result: Result<StoredNodeCount, PublishError>) {
        match result {
            Ok(stored_nodes) => {
                self.success_count += 1;
                self.max_stored_nodes = self.max_stored_nodes.max(stored_nodes);
            }
            Err(error) => *self.error_counts.entry(error).or_default() += 1,
        }
    }

    /// Converts all recorded backend responses into the publish result returned to callers.
    ///
    /// A concurrency conflict blocks success when the number of conflicting
    /// responses is at least the number of successful responses. Otherwise, any
    /// success wins and returns the maximum stored-node count. Without a success,
    /// the most common error is returned, with more specific error categories
    /// preferred when counts tie.
    pub(super) fn into_result(self) -> Result<StoredNodeCount, PublishError> {
        if self.has_conflict_quorum() {
            return Err(PublishError::NotMostRecent);
        }

        if self.success_count > 0 {
            return Ok(self.max_stored_nodes);
        }

        Err(most_common_error(self.error_counts))
    }

    fn has_conflict_quorum(&self) -> bool {
        self.error_counts
            .get(&PublishError::NotMostRecent)
            .is_some_and(|count| *count >= self.success_count)
    }
}

fn most_common_error(error_counts: HashMap<PublishError, usize>) -> PublishError {
    error_counts
        .into_iter()
        .max_by(|(left_error, left_count), (right_error, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| compare_errors(left_error, right_error))
        })
        .map_or(PublishError::UnexpectedResponses, |(error, _)| error)
}

fn compare_errors(left: &PublishError, right: &PublishError) -> Ordering {
    let priority = |error: &PublishError| match error {
        PublishError::UnexpectedResponses => 0,
        PublishError::NoResponses => 1,
        PublishError::NoDhtNodesQueried => 2,
        PublishError::Rejected { .. } => 3,
        PublishError::NotMostRecent => 4,
    };

    priority(left)
        .cmp(&priority(right))
        .then_with(|| match (left, right) {
            (
                PublishError::Rejected {
                    code: left_code,
                    description: left_description,
                },
                PublishError::Rejected {
                    code: right_code,
                    description: right_description,
                },
            ) => left_code
                .cmp(right_code)
                .then_with(|| left_description.cmp(right_description)),
            _ => Ordering::Equal,
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn accumulated_result(
        results: impl IntoIterator<Item = Result<StoredNodeCount, PublishError>>,
    ) -> Result<StoredNodeCount, PublishError> {
        let mut accumulator = PublishResultAccumulator::default();
        for result in results {
            accumulator.record_result(result);
        }
        accumulator.into_result()
    }

    #[rstest]
    #[case::empty(vec![], Err(PublishError::UnexpectedResponses))]
    #[case::maximum_stored_nodes(vec![Ok(3), Ok(11), Ok(7)], Ok(11))]
    #[case::conflict_quorum(
        vec![Ok(11), Err(PublishError::NotMostRecent)],
        Err(PublishError::NotMostRecent)
    )]
    #[case::success_quorum(
        vec![Ok(3), Ok(11), Err(PublishError::NotMostRecent)],
        Ok(11)
    )]
    #[case::conflict_precedes_more_common_error(
        vec![
            Ok(3),
            Ok(5),
            Err(PublishError::NotMostRecent),
            Err(PublishError::NotMostRecent),
            Err(PublishError::NoResponses),
            Err(PublishError::NoResponses),
            Err(PublishError::NoResponses),
        ],
        Err(PublishError::NotMostRecent)
    )]
    #[case::error_priority_breaks_count_tie(
        vec![
            Err(PublishError::NoResponses),
            Err(PublishError::NoDhtNodesQueried),
        ],
        Err(PublishError::NoDhtNodesQueried)
    )]
    #[case::error_count_precedes_priority(
        vec![
            Err(PublishError::NoResponses),
            Err(PublishError::NoResponses),
            Err(PublishError::NoDhtNodesQueried),
        ],
        Err(PublishError::NoResponses)
    )]
    #[case::rejection_details_break_tie(
        vec![
            Err(PublishError::Rejected {
                code: 400,
                description: "first".to_owned(),
            }),
            Err(PublishError::Rejected {
                code: 500,
                description: "second".to_owned(),
            }),
        ],
        Err(PublishError::Rejected {
            code: 500,
            description: "second".to_owned(),
        })
    )]
    fn into_result_applies_publish_selection_rules(
        #[case] results: Vec<Result<StoredNodeCount, PublishError>>,
        #[case] expected: Result<StoredNodeCount, PublishError>,
    ) {
        assert_eq!(accumulated_result(results), expected);
    }
}
