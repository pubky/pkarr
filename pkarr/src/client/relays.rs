use reqwest::Client;
use tokio::runtime::{Builder, Runtime};
use url::Url;

use crate::{Cache, PublicKey, SignedPacket};

use super::shared::{format_url, resolve_from_relay};

pub struct RelaysClient {
    relays: Box<[Url]>,
    http_client: Client,
    cache: Option<Box<dyn Cache>>,
    runtime: Runtime,
}

// TODO: add reqwest client timeout from the ClientBulider timeout settings.
impl RelaysClient {
    pub fn new(relays: Box<[Url]>, cache: Option<Box<dyn Cache>>) -> Self {
        Self {
            relays,
            // TODO: allow passing a runtime.
            runtime: Builder::new_multi_thread()
                .worker_threads(4) // Adjust the number of threads as needed
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime"),
            http_client: Client::default(),
            cache,
        }
    }

    pub fn publish(&self, signed_packet: &SignedPacket, sender: flume::Sender<Result<(), ()>>) {
        let public_key = signed_packet.public_key();
        let path = public_key.to_string();
        let body = signed_packet.to_relay_payload();

        for relay in &self.relays {
            let mut url = relay.clone();
            let mut segments = url.path_segments_mut().unwrap();
            segments.push(&path);
            drop(segments);

            let http_client = self.http_client.clone();
            let body = body.clone();
            let sender = sender.clone();

            self.runtime.spawn(async move {
                let response = http_client.put(url).body(body).send().await;

                match response {
                    Ok(response) => {
                        match response.error_for_status() {
                            Ok(_response) => {
                                // TODO: handle success
                                let _ = sender.send(Ok(()));
                            }
                            Err(_error) => {
                                // TODO: handle error response from a relay.
                            }
                        }
                    }
                    Err(_error) => {
                        // TODO: handle error sending a request to a relay.
                    }
                }
            });
        }
    }

    pub fn resolve(
        &self,
        public_key: &PublicKey,
        sender: flume::Sender<Result<Option<SignedPacket>, ()>>,
    ) {
        let path = public_key.to_string();

        for relay in &self.relays {
            let http_client = self.http_client.clone();
            let relay = relay.clone();
            let sender = sender.clone();
            let cache = self.cache.clone();
            let public_key = public_key.clone();

            self.runtime.spawn(async move {
                match resolve_from_relay(http_client, relay, &public_key, cache).await {
                    Ok(Some(signed_packet)) => {
                        let _ = sender.send(Ok(Some(signed_packet)));
                    }
                    Ok(None) => {}
                    Err(err) => {}
                };
            });
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use std::{thread, time::Duration};
//
//     use pkarr_relay::Relay;
//
//     use super::super::native::*;
//     use crate::{Keypair, SignedPacket};
//
//     #[tokio::test]
//     async fn publish_resolve() {
//         let relay = Relay::start_test().await.unwrap();
//
//         let a = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .build()
//             .unwrap();
//
//         let keypair = Keypair::random();
//
//         let signed_packet = SignedPacket::builder()
//             .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
//             .sign(&keypair)
//             .unwrap();
//
//         a.publish(&signed_packet).await.unwrap();
//
//         let b = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .build()
//             .unwrap();
//
//         let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
//         assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
//
//         let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
//         assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
//         assert_eq!(from_cache.last_seen(), resolved.last_seen());
//     }
//
//     #[tokio::test]
//     async fn thread_safe() {
//         let relay = Relay::start_test().await.unwrap();
//
//         let a = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .build()
//             .unwrap();
//
//         let keypair = Keypair::random();
//
//         let signed_packet = SignedPacket::builder()
//             .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
//             .sign(&keypair)
//             .unwrap();
//
//         a.publish(&signed_packet).await.unwrap();
//
//         let b = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .build()
//             .unwrap();
//
//         tokio::spawn(async move {
//             let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
//             assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
//
//             let from_cache = b.resolve(&keypair.public_key()).await.unwrap().unwrap();
//             assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
//             assert_eq!(from_cache.last_seen(), resolved.last_seen());
//         })
//         .await
//         .unwrap();
//     }
//
//     #[tokio::test]
//     async fn return_expired_packet_fallback() {
//         let relay = Relay::start_test().await.unwrap();
//
//         let client = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .dht_config(mainline::Config {
//                 request_timeout: Duration::from_millis(10),
//                 ..Default::default()
//             })
//             // Everything is expired
//             .maximum_ttl(0)
//             .build()
//             .unwrap();
//
//         let keypair = Keypair::random();
//
//         let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
//
//         client
//             .cache()
//             .unwrap()
//             .put(&keypair.public_key().into(), &signed_packet);
//
//         let resolved = client.resolve(&keypair.public_key()).await.unwrap();
//
//         assert_eq!(resolved, Some(signed_packet));
//     }
//
//     #[tokio::test]
//     async fn ttl_0_test() {
//         let relay = Relay::start_test().await.unwrap();
//
//         let client = Client::builder()
//             .no_default_network()
//             .relays(Some(vec![relay.local_url()]))
//             .maximum_ttl(0)
//             .build()
//             .unwrap();
//
//         let keypair = Keypair::random();
//         let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();
//
//         client.publish(&signed_packet).await.unwrap();
//
//         // First Call
//         let resolved = client
//             .resolve(&signed_packet.public_key())
//             .await
//             .unwrap()
//             .unwrap();
//
//         assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());
//
//         thread::sleep(Duration::from_millis(10));
//
//         let second = client
//             .resolve(&signed_packet.public_key())
//             .await
//             .unwrap()
//             .unwrap();
//         assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
//     }
// }
