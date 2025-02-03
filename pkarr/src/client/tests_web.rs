//! Client wasm/web tests

use wasm_bindgen_test::*;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use crate::{Client, Keypair, SignedPacket};
use url::Url;

#[wasm_bindgen_test]
async fn publish_resolve() {
    console_log::init_with_level(log::Level::Debug).unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    let relays = vec![Url::parse("http://localhost:15411").unwrap()];

    let a = Client::builder().relays(relays.clone()).build().unwrap();
    let b = Client::builder().relays(relays).build().unwrap();

    a.publish(&signed_packet, None).await.unwrap();

    let resolved = b.resolve(&keypair.public_key()).await.unwrap().unwrap();

    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
}

#[wasm_bindgen_test]
async fn not_found() {
    let keypair = Keypair::random();

    let relays = vec![Url::parse("http://localhost:15411").unwrap()];

    let client = Client::builder().relays(relays).build().unwrap();

    let resolved = client.resolve(&keypair.public_key()).await.unwrap();

    assert!(resolved.is_none());
}

#[wasm_bindgen_test]
async fn return_expired_packet_fallback() {
    let keypair = Keypair::random();

    let relays = vec![Url::parse("http://localhost:15411").unwrap()];

    let client = Client::builder()
        .relays(relays)
        .maximum_ttl(0)
        .build()
        .unwrap();

    let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

    client
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &signed_packet);

    let resolved = client.resolve(&keypair.public_key()).await.unwrap();

    assert_eq!(resolved, Some(signed_packet));
}
