use futures_lite::Stream;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::SignedPacket;

use super::PublishError;

/// Run both futures in the background, return the earliest success response,
/// preferring the DHT future if both ready, but complete the other future regardless.
pub async fn spawn_and_select(
    dht_future: impl Future<Output = Result<(), PublishError>> + Send + 'static,
    relays_future: impl Future<Output = Result<(), PublishError>> + Send + 'static,
) -> Result<(), PublishError> {
    let dht_task = tokio::spawn(dht_future);
    let relays_task = tokio::spawn(relays_future);

    futures_lite::future::or(dht_task, relays_task)
        .await
        .expect("tokio join error")
}

/// Run both streams, polling each in a round robin manner,
/// and don't close the stream until both are done.
pub fn select_stream(
    dht_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
    relays_stream: Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
) -> SelectStream {
    SelectStream(false, dht_stream, relays_stream)
}

pub struct SelectStream(
    bool,
    Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
    Pin<Box<dyn Stream<Item = SignedPacket> + Send>>,
);

impl Stream for SelectStream {
    type Item = SignedPacket;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.0 = !this.0;

        let (primary, secondary) = if this.0 {
            (this.1.as_mut(), this.2.as_mut())
        } else {
            (this.2.as_mut(), this.1.as_mut())
        };

        match primary.poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => secondary.poll_next(cx),
            Poll::Pending => Poll::Pending,
        }
    }
}
