use futures_lite::{FutureExt, Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::SignedPacket;

use super::PublishError;

/// Publish to both DHT and configured Relays, resolve after queries on both network are complete.
///
/// If Relays network completes succeeds first, it is possible for the query to the
/// DHT to fail because CAS is outdated, in that case, we should ignore that error.
pub async fn publish_both_networks(
    dht_future: impl Future<Output = Result<(), PublishError>> + Send + 'static,
    relays_future: impl Future<Output = Result<(), PublishError>> + Send + 'static,
) -> Result<(), PublishError> {
    SelectFuture {
        first_result: None,
        dht_future: dht_future.boxed(),
        relays_future: relays_future.boxed(),
    }
    .await
}

pub struct SelectFuture {
    first_result: Option<(Network, Result<(), PublishError>)>,
    dht_future: Pin<Box<dyn Future<Output = Result<(), PublishError>> + Send>>,
    relays_future: Pin<Box<dyn Future<Output = Result<(), PublishError>> + Send>>,
}

impl SelectFuture {
    /// Process a completed future result based on what we've seen before
    fn process_result(
        &mut self,
        network: Network,
        result: Result<(), PublishError>,
    ) -> Poll<Result<(), PublishError>> {
        match &self.first_result {
            // We already have a success on one network, the publish is done.
            Some((_, Ok(()))) => Poll::Ready(Ok(())),
            // We already have a failure, return the later network's result
            Some(_) => Poll::Ready(result),
            // This is our first result, store it and continue polling
            None => {
                self.first_result = Some((network, result));

                Poll::Pending
            }
        }
    }
}

impl Future for SelectFuture {
    type Output = Result<(), PublishError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        let done_network = this.first_result.as_ref().map(|r| r.0);

        // Poll DHT future if not completed yet
        if !matches!(done_network, Some(Network::Dht)) {
            if let Poll::Ready(result) = this.dht_future.as_mut().poll(cx) {
                let poll = this.process_result(Network::Dht, result);
                if poll.is_ready() {
                    return poll;
                }
            }
        }

        // Poll Relays future if not completed yet
        if !matches!(done_network, Some(Network::Relays)) {
            if let Poll::Ready(result) = this.relays_future.as_mut().poll(cx) {
                return this.process_result(Network::Relays, result);
            }
        }

        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{PublishError, QueryError};
    use tokio::sync::oneshot;

    async fn channel_future() -> (
        oneshot::Sender<Result<(), PublishError>>,
        impl Future<Output = Result<(), PublishError>> + Send + 'static,
    ) {
        let (tx, rx) = oneshot::channel();
        (tx, async move { rx.await.unwrap() })
    }

    #[tokio::test]
    async fn relay_success_dht_timeout_returns_ok() {
        let (dht_tx, dht_future) = channel_future().await;
        let (relay_tx, relay_future) = channel_future().await;

        let handle = tokio::spawn(publish_both_networks(dht_future, relay_future));

        dht_tx
            .send(Err(PublishError::Query(QueryError::Timeout)))
            .unwrap();
        relay_tx.send(Ok(())).unwrap();

        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn dht_success_relay_timeout_returns_ok() {
        let (dht_tx, dht_future) = channel_future().await;
        let (relay_tx, relay_future) = channel_future().await;

        let handle = tokio::spawn(publish_both_networks(dht_future, relay_future));

        dht_tx.send(Ok(())).unwrap();
        relay_tx
            .send(Err(PublishError::Query(QueryError::Timeout)))
            .unwrap();

        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn both_fail_returns_error() {
        let (dht_tx, dht_future) = channel_future().await;
        let (relay_tx, relay_future) = channel_future().await;

        let handle = tokio::spawn(publish_both_networks(dht_future, relay_future));

        dht_tx
            .send(Err(PublishError::Query(QueryError::Timeout)))
            .unwrap();
        relay_tx
            .send(Err(PublishError::Query(QueryError::Timeout)))
            .unwrap();

        let result = handle.await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn both_succeed_returns_ok() {
        let (dht_tx, dht_future) = channel_future().await;
        let (relay_tx, relay_future) = channel_future().await;

        let handle = tokio::spawn(publish_both_networks(dht_future, relay_future));

        dht_tx.send(Ok(())).unwrap();
        relay_tx.send(Ok(())).unwrap();

        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

// ==== Stream ====

/// Returns a stream combinator that yields items from two inner streams in a round-robin fashion.
///
/// This combinator will:
/// - Alternate between the two streams on each poll.
/// - Continue polling a stream even after the other is exhausted.
/// - Only terminate when **both** streams have returned `None`.
pub fn select_stream(
    dht_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
    relays_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
) -> SelectStream {
    SelectStream {
        mode: Mode::RoundRobin(Network::Dht),
        dht_stream,
        relays_stream,
    }
}

pub struct SelectStream {
    mode: Mode,
    dht_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
    relays_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
}

#[derive(Clone, Debug)]
enum Mode {
    RoundRobin(Network),
    Exhausted(Network),
}

#[derive(Clone, Copy, Debug)]
enum Network {
    Dht,
    Relays,
}

impl Stream for SelectStream {
    type Item = SignedPacket;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        match this.mode {
            Mode::RoundRobin(current) => {
                // Alternate which stream to poll next
                let (primary, secondary) = match current {
                    Network::Dht => (this.dht_stream.as_mut(), this.relays_stream.as_mut()),
                    Network::Relays => (this.relays_stream.as_mut(), this.dht_stream.as_mut()),
                };

                // Update the Network for next poll
                this.mode = Mode::RoundRobin(match current {
                    Network::Dht => Network::Relays,
                    Network::Relays => Network::Dht,
                });

                // Try polling the current primary stream
                match primary.poll_next(cx) {
                    Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
                    Poll::Ready(None) => {
                        // Primary is exhausted, now only poll the remaining one
                        this.mode = Mode::Exhausted(current);
                        secondary.poll_next(cx)
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            Mode::Exhausted(exhausted) => match exhausted {
                Network::Dht => this.relays_stream.poll_next(cx),
                Network::Relays => this.dht_stream.poll_next(cx),
            },
        }
    }
}
