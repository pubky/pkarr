//! Service requests, bundle acceptance, and cache updates.

#![cfg(feature = "service-client")]

use base64::{engine::general_purpose::STANDARD, Engine};
use pkarr::{dns::rdata::SVCB, Keypair, SignedPacket};
use pkresolve::{
    service_client::{Error, Preparation},
    HttpProtocol, ResolutionRequest, ServiceClient,
};
use serde_json::json;

fn request(key: &Keypair) -> ResolutionRequest {
    ResolutionRequest::new(
        format!("app.{}", key.to_z32()).parse().unwrap(),
        8443,
        vec![HttpProtocol::Http2, HttpProtocol::Http11],
    )
    .unwrap()
}

fn packet(key: &Keypair, address: &str) -> SignedPacket {
    SignedPacket::builder()
        .https(
            "_8443._https.app".try_into().unwrap(),
            SVCB::new(1, ".".try_into().unwrap()),
            300,
        )
        .address(
            "_8443._https.app".try_into().unwrap(),
            address.parse().unwrap(),
            300,
        )
        .sign(key)
        .unwrap()
}

fn bundle(packets: &[SignedPacket]) -> Vec<u8> {
    let packets: Vec<_> = packets
        .iter()
        .map(|packet| STANDARD.encode(packet.serialize()))
        .collect();
    serde_json::to_vec(&json!({ "schema": 1, "packets": packets })).unwrap()
}

#[test]
fn service_request_preserves_scope_port_and_protocol_preferences() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let mut client = ServiceClient::default();
    let original = request(&key);
    let Preparation::Fetch(fetch) = client.prepare(original.clone()).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(fetch.body()).unwrap(),
        json!({
            "service": original.service().as_str(),
            "url_port": 8443,
            "supported_alpn": ["h2", "http/1.1"],
        })
    );

    let response = bundle(&[packet(&key, "203.0.113.10")]);
    let resolved = client.accept_bundle(fetch, &response).unwrap();
    assert_eq!(&resolved.service, original.service());
    assert_eq!(resolved.url_port, 8443);
    let Preparation::Ready(cached) = client.prepare(original).unwrap() else {
        panic!("accepted bundle should populate the cache");
    };
    assert_eq!(cached.candidates, resolved.candidates);
}

#[test]
fn bundle_for_another_service_does_not_satisfy_or_populate_the_cache() {
    let alice = Keypair::from_secret_key(&[1; 32]);
    let bob = Keypair::from_secret_key(&[2; 32]);
    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request(&alice)).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    let response = bundle(&[packet(&bob, "203.0.113.20")]);
    assert!(matches!(
        client.accept_bundle(fetch, &response),
        Err(Error::Resolution(pkresolve::Error::MissingPacket(key)))
            if *key == alice.public_key()
    ));
    assert!(matches!(
        client.prepare(request(&bob)).unwrap(),
        Preparation::Fetch(_)
    ));
}

#[test]
fn duplicate_packets_leave_previously_cached_routes_unchanged() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let mut client = ServiceClient::default();
    let Preparation::Fetch(first) = client.prepare(request(&key)).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    // Both requests are prepared before either response arrives.
    let Preparation::Fetch(second) = client.prepare(request(&key)).unwrap() else {
        panic!("pending fetch should not populate the cache");
    };
    let accepted = client
        .accept_bundle(first, &bundle(&[packet(&key, "203.0.113.10")]))
        .unwrap();
    let conflicting = bundle(&[packet(&key, "203.0.113.20"), packet(&key, "203.0.113.30")]);
    assert!(matches!(
        client.accept_bundle(second, &conflicting),
        Err(Error::DuplicatePacket(owner)) if owner == key.to_z32()
    ));
    let Preparation::Ready(cached) = client.prepare(request(&key)).unwrap() else {
        panic!("failed response should preserve the cached resolution");
    };
    assert_eq!(cached.candidates, accepted.candidates);
}
