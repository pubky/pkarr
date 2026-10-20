//! JSON compatibility and validation at the shared protocol boundary.

#![cfg(feature = "service-protocol")]

use pkarr::Keypair;
use pkresolve::{
    service_protocol::{BundleResolution, ResolveBundle, BUNDLE_SCHEMA},
    EndpointCandidate, HttpProtocol, Resolution, ResolutionRequest,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

fn request() -> Value {
    json!({
        "service": format!("app.{}", Keypair::from_secret_key(&[1; 32]).to_z32()),
        "url_port": 8443,
        "supported_alpn": ["h2", "http/1.1"],
    })
}

#[test]
fn validation_preserves_protocol_order_and_normalizes_service() {
    let original = request();
    let mut input = original.clone();
    input["service"] = json!(format!(
        "{}.",
        original["service"].as_str().unwrap().to_ascii_uppercase()
    ));
    input["supported_alpn"] = json!(["h2", "http/1.1", "h2"]);
    let validated: ResolutionRequest = serde_json::from_value(input).unwrap();
    assert_eq!(
        validated.supported_protocols(),
        &[HttpProtocol::Http2, HttpProtocol::Http11]
    );
    assert_eq!(serde_json::to_value(&validated).unwrap(), original);
    let constructed = ResolutionRequest::new(
        validated.service().clone(),
        validated.url_port(),
        vec![
            HttpProtocol::Http2,
            HttpProtocol::Http11,
            HttpProtocol::Http2,
        ],
    )
    .unwrap();
    assert_eq!(
        constructed.supported_protocols(),
        validated.supported_protocols()
    );
    assert_eq!(serde_json::to_value(constructed).unwrap(), original);
}

#[test]
fn deserialization_rejects_invalid_requests() {
    for (field, value) in [
        ("service", json!("https://invalid")),
        ("url_port", json!(0)),
        ("url_port", json!(-1)),
        ("url_port", json!(65536)),
        ("supported_alpn", json!([])),
        ("supported_alpn", json!(["h3"])),
    ] {
        let mut invalid = request();
        invalid[field] = value;
        assert!(serde_json::from_value::<ResolutionRequest>(invalid).is_err());
    }

    let mut unknown = request();
    unknown["unknown"] = json!(true);
    assert!(serde_json::from_value::<ResolutionRequest>(unknown).is_err());
    for field in ["service", "url_port", "supported_alpn"] {
        let mut missing = request();
        missing.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<ResolutionRequest>(missing).is_err());
        let mut wrong_type = request();
        wrong_type[field] = json!(false);
        assert!(serde_json::from_value::<ResolutionRequest>(wrong_type).is_err());
    }
}

#[test]
fn bundle_preserves_wire_format_and_ignores_unknown_fields() {
    let original = json!({
        "schema": 1,
        "packets": ["first", "second"],
        "resolution": {
            "service": request()["service"],
            "url_port": 8443,
            "candidates": [],
            "valid_for_ms": 0,
        },
    });
    let mut input = original.clone();
    input["unknown"] = json!(true);
    let bundle: ResolveBundle = serde_json::from_value(input).unwrap();
    assert_eq!(bundle.schema, BUNDLE_SCHEMA);
    assert_eq!(bundle.packets, ["first", "second"]);
    assert_eq!(bundle.resolution.url_port, 8443);
    assert_eq!(serde_json::to_value(bundle).unwrap(), original);
}

#[test]
fn summary_encodes_routes_and_clamps_expired_lifetime_to_zero() {
    let request: ResolutionRequest = serde_json::from_value(request()).unwrap();
    let resolution = Resolution {
        service: request.service().clone(),
        url_port: request.url_port(),
        candidates: vec![EndpointCandidate {
            addresses: vec![
                "203.0.113.10".parse().unwrap(),
                "2001:db8::1".parse().unwrap(),
            ],
            port: 9443.try_into().unwrap(),
            priority: 1,
            protocols: vec![HttpProtocol::Http2, HttpProtocol::Http11],
        }],
        expires_at: Instant::now() - Duration::from_secs(1),
    };
    let summary = BundleResolution::from(resolution);
    let json = serde_json::to_value(&summary).unwrap();
    assert_eq!(
        json,
        json!({
            "service": request.service().as_str(),
            "url_port": 8443,
            "candidates": [{
                "addresses": ["203.0.113.10", "2001:db8::1"],
                "port": 9443,
                "priority": 1,
                "protocols": ["h2", "http/1.1"],
            }],
            "valid_for_ms": 0,
        })
    );
    let decoded: BundleResolution = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.candidates, summary.candidates);
}
