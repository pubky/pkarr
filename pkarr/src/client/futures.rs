use std::future::Future;

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
