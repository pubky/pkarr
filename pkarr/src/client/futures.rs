use futures_lite::{FutureExt, Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::client::ConcurrencyError;
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
        match self.first_result {
            // We already have a success, ignore CAS failures
            Some((_, Ok(()))) => Poll::Ready(match result {
                Err(PublishError::Concurrency(ConcurrencyError::CasFailed)) => Ok(()),
                _ => result,
            }),
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
                return this.process_result(Network::Dht, result);
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
                // The non-primary network this round.
                let other = match current {
                    Network::Dht => Network::Relays,
                    Network::Relays => Network::Dht,
                };

                let (primary, secondary) = match current {
                    Network::Dht => (this.dht_stream.as_mut(), this.relays_stream.as_mut()),
                    Network::Relays => (this.relays_stream.as_mut(), this.dht_stream.as_mut()),
                };

                // Alternate priority on the next poll.
                this.mode = Mode::RoundRobin(other);

                // Poll the primary first, but fall back to the secondary when it
                // is pending so a ready secondary is never starved behind it.
                match primary.poll_next(cx) {
                    Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
                    Poll::Ready(None) => {
                        this.mode = Mode::Exhausted(current);
                        secondary.poll_next(cx)
                    }
                    Poll::Pending => match secondary.poll_next(cx) {
                        // Secondary exhausted: keep the pending primary alive.
                        Poll::Ready(None) => {
                            this.mode = Mode::Exhausted(other);
                            Poll::Pending
                        }
                        poll => poll,
                    },
                }
            }
            Mode::Exhausted(exhausted) => match exhausted {
                Network::Dht => this.relays_stream.poll_next(cx),
                Network::Relays => this.dht_stream.poll_next(cx),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::task::{Context, Poll, Waker};

    use futures_lite::stream;

    use crate::{Keypair, PublicKey, SignedPacket};

    type BoxStream = Pin<Box<dyn Stream<Item = SignedPacket> + Send>>;

    fn packet() -> SignedPacket {
        SignedPacket::builder().sign(&Keypair::random()).unwrap()
    }

    fn pending() -> BoxStream {
        Box::pin(stream::pending())
    }

    fn ready(packet: SignedPacket) -> BoxStream {
        Box::pin(stream::once(packet))
    }

    /// A ready stream must be yielded on the first poll even when the
    /// round-robin primary is pending. Regression test for a starvation bug
    /// where `SelectStream` returned `Pending` without polling the secondary,
    /// so a fast relay response sat unyielded until the slow DHT woke the task.
    fn assert_yields(mut select: SelectStream, expected: PublicKey) {
        let mut cx = Context::from_waker(Waker::noop());
        match Pin::new(&mut select).poll_next(&mut cx) {
            Poll::Ready(Some(got)) => assert_eq!(got.public_key(), expected),
            Poll::Ready(None) => panic!("stream terminated unexpectedly"),
            Poll::Pending => panic!("ready secondary starved behind a pending primary"),
        }
    }

    #[test]
    fn ready_relay_not_starved_by_pending_dht() {
        // Starts in RoundRobin(Dht): DHT (primary) pending, relay ready.
        let p = packet();
        let expected = p.public_key();
        assert_yields(select_stream(pending(), ready(p)), expected);
    }

    #[test]
    fn ready_dht_not_starved_by_pending_relay() {
        // Symmetric: round-robin pointer on Relays, relay (primary) pending.
        let p = packet();
        let expected = p.public_key();
        assert_yields(
            SelectStream {
                mode: Mode::RoundRobin(Network::Relays),
                dht_stream: ready(p),
                relays_stream: pending(),
            },
            expected,
        );
    }
}
