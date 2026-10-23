//! Service requests, bundle acceptance, and cache updates.

#![cfg(feature = "service-client")]

use base64::{engine::general_purpose::STANDARD, Engine};
use pkarr::{dns::rdata::SVCB, Keypair, SignedPacket};
use pkresolve::{
    service_client::{Error, Preparation},
    service_protocol::{ResolveBundle, BUNDLE_SCHEMA},
    HttpProtocol, ResolutionRequest, ServiceClient,
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};

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
    let first = &packets[0];
    let request = ResolutionRequest::new(
        format!("app.{}", first.public_key()).parse().unwrap(),
        8443,
        vec![HttpProtocol::Http2, HttpProtocol::Http11],
    )
    .unwrap();
    let provider = HashMap::from([(first.public_key(), Arc::new(first.clone()))]);
    let resolution = pkresolve::derive_resolution(&request, &provider).unwrap();
    let packets: Vec<_> = packets
        .iter()
        .map(|packet| STANDARD.encode(packet.serialize()))
        .collect();
    serde_json::to_vec(&ResolveBundle {
        schema: BUNDLE_SCHEMA,
        packets,
        resolution: resolution.into(),
    })
    .unwrap()
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

    let validated: ResolutionRequest = serde_json::from_str(fetch.body()).unwrap();
    assert_eq!(validated.service(), original.service());
    assert_eq!(validated.url_port(), original.url_port());
    assert_eq!(
        validated.supported_protocols(),
        original.supported_protocols()
    );

    let response = bundle(&[packet(&key, "203.0.113.10")]);
    let resolved = client.accept_bundle(fetch, &response).unwrap();
    assert_eq!(resolved.tls_public_key, key.public_key());
    assert_eq!(&resolved.service, original.service());
    assert_eq!(resolved.url_port, 8443);
    let Preparation::Ready(cached) = client.prepare(original).unwrap() else {
        panic!("accepted bundle should populate the cache");
    };
    assert_eq!(cached.candidates, resolved.candidates);
}

#[test]
fn missing_or_null_summary_is_rejected_without_populating_cache() {
    let key = Keypair::from_secret_key(&[1; 32]);
    for omit_summary in [true, false] {
        let mut response: serde_json::Value =
            serde_json::from_slice(&bundle(&[packet(&key, "203.0.113.10")])).unwrap();
        if omit_summary {
            response.as_object_mut().unwrap().remove("resolution");
        } else {
            response["resolution"] = serde_json::Value::Null;
        }
        let mut client = ServiceClient::default();
        let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
            panic!("empty cache should require a bundle");
        };
        assert!(matches!(
            client.accept_bundle(fetch, &serde_json::to_vec(&response).unwrap()),
            Err(Error::Json(_))
        ));
        assert!(matches!(
            client.prepare(request(&key)).unwrap(),
            Preparation::Fetch(_)
        ));
    }
}

#[test]
fn summary_cannot_override_routes_or_replace_packet_evidence() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let other_key = Keypair::from_secret_key(&[2; 32]);
    let mut response: serde_json::Value =
        serde_json::from_slice(&bundle(&[packet(&key, "203.0.113.10")])).unwrap();
    response["resolution"] = json!({
        "tls_public_key": other_key.to_z32(),
        "service": other_key.to_z32(),
        "url_port": 1234,
        "candidates": [{
            "addresses": ["203.0.113.99"],
            "port": 1234,
            "priority": 0,
            "protocols": ["h2"],
        }],
        "valid_until_unix_ms": u64::MAX,
    });
    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    let resolved = client
        .accept_bundle(fetch, &serde_json::to_vec(&response).unwrap())
        .unwrap();
    assert_eq!(resolved.service, *request(&key).service());
    assert_eq!(resolved.tls_public_key, key.public_key());
    assert_eq!(resolved.url_port, 8443);
    assert_eq!(
        resolved.candidates[0].addresses[0].to_string(),
        "203.0.113.10"
    );
    assert_eq!(resolved.candidates[0].port.get(), 8443);
    assert!(
        resolved
            .expires_at
            .saturating_duration_since(std::time::Instant::now())
            <= std::time::Duration::from_secs(300)
    );

    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    response["packets"] = json!([]);
    assert!(matches!(
        client.accept_bundle(fetch, &serde_json::to_vec(&response).unwrap()),
        Err(Error::Resolution(pkresolve::Error::MissingPacket(_)))
    ));
}

#[test]
fn unsupported_bundle_schema_does_not_populate_the_cache() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    let mut response: ResolveBundle =
        serde_json::from_slice(&bundle(&[packet(&key, "203.0.113.10")])).unwrap();
    response.schema = 99;
    assert!(matches!(
        client.accept_bundle(fetch, &serde_json::to_vec(&response).unwrap()),
        Err(Error::UnsupportedSchema(99))
    ));
    assert!(matches!(
        client.prepare(request(&key)).unwrap(),
        Preparation::Fetch(_)
    ));
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

#[test]
fn tampered_packets_leave_existing_cache_unchanged() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let other = Keypair::from_secret_key(&[2; 32]);
    let replacement = packet(&key, "203.0.113.20");
    let signed = packet(&other, "203.0.113.30");
    // Native layout: observation(8), key(32), signature(64), timestamp(8), DNS.
    // The last DNS byte is part of the final A record's address.
    for offset in [8, 40, 104, signed.serialize().len() - 1] {
        let mut client = ServiceClient::default();
        let Preparation::Fetch(first) = client.prepare(request(&key)).unwrap() else {
            panic!("expected initial fetch");
        };
        let Preparation::Fetch(pending) = client.prepare(request(&key)).unwrap() else {
            panic!("expected pending fetch");
        };
        let accepted = client
            .accept_bundle(first, &bundle(&[packet(&key, "203.0.113.10")]))
            .unwrap();
        let mut response: ResolveBundle =
            serde_json::from_slice(&bundle(&[replacement.clone(), signed.clone()])).unwrap();
        let mut bytes = signed.serialize();
        if offset == 8 {
            bytes[8..40]
                .copy_from_slice(Keypair::from_secret_key(&[3; 32]).public_key().as_bytes());
        } else {
            bytes[offset] ^= 1;
        }
        response.packets[1] = STANDARD.encode(bytes);
        assert!(
            matches!(
                client.accept_bundle(pending, &serde_json::to_vec(&response).unwrap()),
                Err(Error::Verification(_))
            ),
            "tampering at byte {offset} must fail verification"
        );
        let Preparation::Ready(cached) = client.prepare(request(&key)).unwrap() else {
            panic!("failed response must preserve cache");
        };
        assert_eq!(cached.candidates, accepted.candidates);
        assert!(matches!(
            client.prepare(request(&other)).unwrap(),
            Preparation::Fetch(_)
        ));
    }
}

#[test]
fn signature_verification_preserves_unsigned_observation_time() {
    use std::time::{Duration, Instant, SystemTime};

    let key = Keypair::from_secret_key(&[1; 32]);
    let mut signed = packet(&key, "203.0.113.10");
    let signature = signed.signature();
    signed.set_last_seen(&(SystemTime::now() - Duration::from_secs(120)).into());
    let restored = SignedPacket::deserialize(&signed.serialize()).unwrap();
    restored.verify_signature().unwrap();
    assert_eq!(restored.signature(), signature);
    assert_eq!(restored.last_seen(), signed.last_seen());

    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
        panic!("expected fetch");
    };
    let resolved = client.accept_bundle(fetch, &bundle(&[signed])).unwrap();
    let remaining = resolved
        .expires_at
        .saturating_duration_since(Instant::now());
    assert!(!remaining.is_zero());
    assert!(remaining <= Duration::from_secs(180));
}

#[test]
fn truncated_native_packets_are_rejected_without_panicking() {
    let key = Keypair::from_secret_key(&[1; 32]);
    let signed = packet(&key, "203.0.113.10");
    let bytes = signed.serialize();
    let mut response: ResolveBundle = serde_json::from_slice(&bundle(&[signed])).unwrap();
    for length in 0..bytes.len() {
        let mut client = ServiceClient::default();
        let Preparation::Fetch(fetch) = client.prepare(request(&key)).unwrap() else {
            panic!("expected fetch");
        };
        response.packets[0] = STANDARD.encode(&bytes[..length]);
        assert!(client
            .accept_bundle(fetch, &serde_json::to_vec(&response).unwrap())
            .is_err());
    }
}
