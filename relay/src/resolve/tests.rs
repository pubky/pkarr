use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    num::NonZeroU32,
    sync::Arc,
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    extract::ConnectInfo,
    response::{IntoResponse, Response},
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use http::{
    header::{CACHE_CONTROL, CONTENT_TYPE},
    Request, StatusCode,
};
use pkarr::{
    dht::{DhtClient, DhtConfig, ReportPolicy},
    dns::rdata::SVCB,
    extra::lmdb_cache::LmdbCache,
    Cache, CacheKey, Keypair, SignedPacket, Timestamp,
};
use pkresolve::{
    service_client::Preparation,
    service_protocol::{ResolveBundle, BUNDLE_SCHEMA},
    AsyncPacketProvider, HttpProtocol, ResolutionRequest, ServiceClient,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use super::{
    packet_provider::{RelayPacketProvider, MAX_RESOLUTION_PACKETS},
    resolution_error,
};
use crate::{
    build_rate_limiters, create_app, dht_service::DhtService, error::Error, real_ip::RealIp,
    to_socket_address_v4, AppState, RateLimiterConfig,
};

struct TestRelay {
    app: Router,
    cache: Arc<LmdbCache>,
    dht: DhtClient,
    service: DhtService,
    _testnet: mainline::Testnet,
    _storage: tempfile::TempDir,
}

impl TestRelay {
    async fn new(limits: Option<RateLimiterConfig>) -> Self {
        let testnet = mainline::Testnet::builder(2).build().unwrap();
        let storage = tempfile::tempdir().unwrap();
        let cache = Arc::new(unsafe { LmdbCache::open(storage.path(), 20).unwrap() });
        let mut config = DhtConfig::default();
        config.bootstrap = Some(to_socket_address_v4(&testnet.bootstrap));
        config.bind_address = Some(Ipv4Addr::LOCALHOST);
        config.request_timeout = Duration::from_millis(100);
        let limits = build_rate_limiters(limits.as_ref(), &mut config)
            .await
            .unwrap();
        let dht = DhtClient::build(config).unwrap();
        let service = DhtService::new(
            300,
            86400,
            Arc::clone(&cache),
            dht.clone(),
            ReportPolicy::testnet(),
            limits.user_dht,
        );
        let app = create_app(
            AppState {
                dht: service.clone(),
            },
            limits.http,
            limits.behind_proxy,
        );
        Self {
            app,
            cache,
            dht,
            service,
            _testnet: testnet,
            _storage: storage,
        }
    }

    fn cache(&self, packet: &SignedPacket) {
        self.cache
            .put(&CacheKey::from(&packet.public_key()), packet);
    }

    async fn post(&self, body: impl Into<Body>) -> Response {
        self.post_with_content_type(body, Some("application/json"))
            .await
    }

    async fn post_with_content_type(
        &self,
        body: impl Into<Body>,
        content_type: Option<&str>,
    ) -> Response {
        let mut builder = Request::post("/v1/resolve");
        if let Some(content_type) = content_type {
            builder = builder.header(CONTENT_TYPE, content_type);
        }
        let mut request = builder.body(body.into()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 1234))));
        self.app.clone().oneshot(request).await.unwrap()
    }
}

fn packet(keypair: &Keypair, ttl: u32) -> SignedPacket {
    SignedPacket::builder()
        .https(
            "_8443._https.app".try_into().unwrap(),
            SVCB::new(1, ".".try_into().unwrap()),
            ttl,
        )
        .address(
            "_8443._https.app".try_into().unwrap(),
            "203.0.113.10".parse().unwrap(),
            ttl,
        )
        .sign(keypair)
        .unwrap()
}

fn request(keypair: &Keypair) -> ResolutionRequest {
    ResolutionRequest::new(
        format!("app.{}", keypair.to_z32()).parse().unwrap(),
        8443,
        vec![HttpProtocol::Http2, HttpProtocol::Http11],
    )
    .unwrap()
}

fn body(keypair: &Keypair) -> Value {
    json!({"service": format!("app.{}", keypair.to_z32()), "url_port": 8443, "supported_alpn": ["h2", "http/1.1"]})
}

#[tokio::test]
async fn provider_errors_preserve_status_headers_and_detail() {
    let error = pkresolve::Error::Provider {
        key: Box::new(Keypair::random().public_key()),
        source: Box::new(Error::invalid_signed_packet(42)),
    };
    let response = resolution_error(error).into_response();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        response.headers()[pkarr::PKARR_INVALID_SIGNED_PACKET_SEQ],
        "42"
    );
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(
        bytes.as_ref(),
        b"DHT mutable item at seq 42 is not a valid signed packet"
    );

    let error = pkresolve::Error::Provider {
        key: Box::new(Keypair::random().public_key()),
        source: Box::new(std::io::Error::other("packet source unavailable")),
    };
    let response = resolution_error(error).into_response();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(bytes.as_ref(), b"packet source unavailable");
}

#[tokio::test]
async fn cached_bundle_round_trips_through_service_client_without_refreshing_observation() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let mut packet = packet(&keypair, 300);
    let observed = Timestamp::from(Timestamp::now().as_u64() - 30_000_000);
    packet.set_last_seen(&observed);
    relay.cache(&packet);

    let mut client = ServiceClient::default();
    let original = request(&keypair);
    let Preparation::Fetch(fetch) = client.prepare(original.clone()).unwrap() else {
        panic!("empty cache");
    };
    let response = relay.post(fetch.body().to_owned()).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
    assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    let bundle: ResolveBundle = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(bundle.schema, BUNDLE_SCHEMA);
    assert_eq!(bundle.packets.len(), 1);
    let encoded = &bundle.packets[0];
    assert_eq!(STANDARD.decode(encoded).unwrap(), packet.serialize());

    let resolution = client.accept_bundle(fetch, &bytes).unwrap();
    let summary = &bundle.resolution;
    assert_eq!(summary.service, original.service().as_str());
    assert_eq!(summary.url_port, original.url_port());
    assert_eq!(summary.candidates, resolution.candidates);
    let valid_for_ms = summary.valid_for_ms;
    assert!(valid_for_ms > 0 && valid_for_ms <= 270_000);
    assert_eq!(resolution.service, *original.service());
    assert_eq!(resolution.url_port, 8443);
    assert_eq!(resolution.candidates[0].port.get(), 8443);
    assert_eq!(
        resolution.candidates[0].addresses,
        vec!["203.0.113.10".parse::<IpAddr>().unwrap()]
    );
    assert!(matches!(
        client.prepare(original).unwrap(),
        Preparation::Ready(_)
    ));
}

#[tokio::test]
async fn cache_miss_fetches_from_dht_and_populates_shared_cache() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let packet = packet(&keypair, 300);
    relay.dht.publish(&packet).await.unwrap();
    let cache_key = CacheKey::from(&keypair.public_key());
    assert!(relay.cache.get(&cache_key).is_none());

    let response = relay.post(body(&keypair).to_string()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let cached = relay.cache.get(&cache_key).unwrap();
    assert_eq!(cached.as_bytes(), packet.as_bytes());
}

#[tokio::test]
async fn expired_signed_ttl_does_not_override_relay_minimum_ttl() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let mut packet = packet(&keypair, 60);
    relay.dht.publish(&packet).await.unwrap();
    let stale_observation = Timestamp::from(Timestamp::now().as_u64() - 120_000_000);
    packet.set_last_seen(&stale_observation);
    relay.cache(&packet);

    let response = relay.post(body(&keypair).to_string()).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "expired record"
    );
    let cached = relay
        .cache
        .get(&CacheKey::from(&keypair.public_key()))
        .unwrap();
    assert_eq!(*cached.last_seen(), stale_observation);
}

#[tokio::test]
async fn invalid_requests_are_rejected_before_lookup() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    for content_type in [None, Some("text/plain")] {
        assert_eq!(
            relay
                .post_with_content_type(body(&keypair).to_string(), content_type)
                .await
                .status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }
    for (field, value) in [
        ("service", json!("https://invalid")),
        ("url_port", json!(0)),
        ("supported_alpn", json!([])),
        ("supported_alpn", json!(["h3"])),
    ] {
        let mut invalid = body(&keypair);
        invalid[field] = value;
        assert_eq!(
            relay.post(invalid.to_string()).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(relay.post("{").await.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        relay.post(" ".repeat(1105)).await.status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let mut unknown = body(&keypair);
    unknown["allowed_authentication"] = json!(["pubky_raw_key"]);
    assert_eq!(
        relay.post(unknown.to_string()).await.status(),
        StatusCode::BAD_REQUEST
    );
    for field in ["service", "url_port", "supported_alpn"] {
        let mut missing = body(&keypair);
        missing.as_object_mut().unwrap().remove(field);
        assert_eq!(
            relay.post(missing.to_string()).await.status(),
            StatusCode::BAD_REQUEST
        );
        let mut wrong_type = body(&keypair);
        wrong_type[field] = json!(false);
        assert_eq!(
            relay.post(wrong_type.to_string()).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
}

#[tokio::test]
async fn absent_packet_and_incompatible_service_are_distinct_failures() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let response = relay.post(body(&keypair).to_string()).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // An address alone cannot authorize a route without HTTPS ServiceMode.
    let address_only = SignedPacket::builder()
        .address(
            "_8443._https.app".try_into().unwrap(),
            "203.0.113.10".parse().unwrap(),
            300,
        )
        .sign(&keypair)
        .unwrap();
    relay.cache(&address_only);
    let response = relay.post(body(&keypair).to_string()).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "no compatible endpoint"
    );
}

#[tokio::test]
async fn direct_service_with_alternate_fallback_returns_bundle() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let mut direct = SVCB::new(1, ".".try_into().unwrap());
    direct.set_port(6287);
    let address: Ipv4Addr = "34.65.156.171".parse().unwrap();
    direct.set_ipv4hint(&[address.into()]);
    let packet = SignedPacket::builder()
        .https(".".try_into().unwrap(), direct, 3600)
        .https(
            ".".try_into().unwrap(),
            SVCB::new(10, "homeserver.pubky.app".try_into().unwrap()),
            3600,
        )
        .address(".".try_into().unwrap(), address.into(), 3600)
        .sign(&keypair)
        .unwrap();
    relay.cache(&packet);
    let request = ResolutionRequest::new(
        keypair.to_z32().parse().unwrap(),
        443,
        vec![HttpProtocol::Http2, HttpProtocol::Http11],
    )
    .unwrap();
    let mut client = ServiceClient::default();
    let Preparation::Fetch(fetch) = client.prepare(request).unwrap() else {
        panic!("empty cache should require a bundle");
    };
    let response = relay.post(fetch.body().to_owned()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    let bundle: ResolveBundle = serde_json::from_slice(&bytes).unwrap();
    let result = client.accept_bundle(fetch, &bytes).unwrap();
    assert_eq!(bundle.resolution.candidates, result.candidates);
    assert_eq!(result.service.public_key(), &keypair.public_key());
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.candidates[0].port.get(), 6287);
    assert_eq!(result.candidates[0].addresses, vec![IpAddr::V4(address)]);
}

#[tokio::test]
async fn unsupported_routing_does_not_return_a_bundle() {
    let relay = TestRelay::new(None).await;
    let keypair = Keypair::random();
    let alias = SignedPacket::builder()
        .https(
            "_8443._https.app".try_into().unwrap(),
            SVCB::new(0, "elsewhere.example".try_into().unwrap()),
            300,
        )
        .sign(&keypair)
        .unwrap();
    relay.cache(&alias);
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn dht_quota_blocks_misses_and_expired_packets_but_allows_fresh_cache_hits() {
    let relay = TestRelay::new(Some(RateLimiterConfig {
        quota: "100r/s".parse().unwrap(),
        burst: NonZeroU32::new(100),
        user_dht_quota: "1r/h".parse().unwrap(),
        user_dht_burst: NonZeroU32::new(1),
        ..RateLimiterConfig::default()
    }))
    .await;
    let keypair = Keypair::random();
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::TOO_MANY_REQUESTS
    );

    let mut packet = packet(&keypair, 60);
    relay.cache(&packet);
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::OK
    );

    packet.set_last_seen(&Timestamp::from(Timestamp::now().as_u64() - 600_000_000));
    relay.cache(&packet);
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn http_quota_also_applies_to_cached_bundles() {
    let relay = TestRelay::new(Some(RateLimiterConfig {
        quota: "1r/h".parse().unwrap(),
        burst: NonZeroU32::new(1),
        ..RateLimiterConfig::default()
    }))
    .await;
    let keypair = Keypair::random();
    relay.cache(&packet(&keypair, 300));
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        relay.post(body(&keypair).to_string()).await.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn multiple_owners_use_shared_cache_and_produce_one_bundle() {
    let relay = TestRelay::new(Some(RateLimiterConfig {
        user_dht_quota: "1r/h".parse().unwrap(),
        user_dht_burst: NonZeroU32::new(1),
        ..RateLimiterConfig::default()
    }))
    .await;
    let cached_owner = Keypair::random();
    let network_owner = Keypair::random();
    let mut cached_packet = packet(&cached_owner, 300);
    cached_packet.set_last_seen(&Timestamp::from(Timestamp::now().as_u64() - 30_000_000));
    relay.cache(&cached_packet);
    relay
        .dht
        .publish(&packet(&network_owner, 300))
        .await
        .unwrap();

    let provider =
        RelayPacketProvider::new(&relay.service, Some(RealIp(Ipv4Addr::LOCALHOST.into())));
    let cached_key = cached_owner.public_key();
    let network_key = network_owner.public_key();
    let (cached, network, repeated) = tokio::join!(
        provider.get_packet(&cached_key),
        provider.get_packet(&network_key),
        provider.get_packet(&network_key),
    );
    let cached = cached.unwrap().unwrap();
    let network = network.unwrap().unwrap();
    assert!(Arc::ptr_eq(&network, &repeated.unwrap().unwrap()));
    assert_eq!(cached.serialize(), cached_packet.serialize());
    assert_eq!(
        relay
            .cache
            .get(&CacheKey::from(&network_key))
            .unwrap()
            .as_bytes(),
        network.as_bytes()
    );

    let resolution = pkresolve::derive_resolution_async(&request(&cached_owner), &provider)
        .await
        .unwrap();
    let bundle = provider.into_bundle(resolution).unwrap();
    assert_eq!(bundle.packets.len(), 2);
    for packet in [&cached, &network] {
        assert!(bundle
            .packets
            .contains(&STANDARD.encode(packet.serialize())));
    }
    let bytes = serde_json::to_vec(&bundle).unwrap();
    // The actual client accepts the multi-owner bundle for either service.
    for owner in [&cached_owner, &network_owner] {
        let mut client = ServiceClient::default();
        let Preparation::Fetch(fetch) = client.prepare(request(owner)).unwrap() else {
            panic!("empty cache");
        };
        assert_eq!(
            client
                .accept_bundle(fetch, &bytes)
                .unwrap()
                .service
                .public_key(),
            &owner.public_key()
        );
    }
}

#[tokio::test]
async fn traversal_retains_original_versions_while_new_requests_see_cache_updates() {
    let relay = TestRelay::new(None).await;
    let first_owner = Keypair::random();
    let second_owner = Keypair::random();
    let original = packet(&first_owner, 300);
    relay.cache(&original);
    let provider = RelayPacketProvider::new(&relay.service, None);
    let first = provider
        .get_packet(&first_owner.public_key())
        .await
        .unwrap()
        .unwrap();

    let newer = packet(&first_owner, 600);
    assert!(newer.more_recent_than(&original));
    relay.cache(&newer);
    relay.cache(&packet(&second_owner, 300));
    provider
        .get_packet(&second_owner.public_key())
        .await
        .unwrap();
    let repeated = provider
        .get_packet(&first_owner.public_key())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&first, &repeated));
    let resolution = pkresolve::derive_resolution_async(&request(&first_owner), &provider)
        .await
        .unwrap();
    let bundle = provider.into_bundle(resolution).unwrap();
    assert_eq!(bundle.packets.len(), 2);
    assert!(bundle
        .packets
        .contains(&STANDARD.encode(original.serialize())));
    assert!(!bundle.packets.contains(&STANDARD.encode(newer.serialize())));

    let next_request = RelayPacketProvider::new(&relay.service, None);
    let latest = next_request
        .get_packet(&first_owner.public_key())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(latest.serialize(), newer.serialize());
}

#[tokio::test]
async fn failed_lookup_is_retained_and_prevents_partial_bundle() {
    let relay = TestRelay::new(Some(RateLimiterConfig {
        user_dht_quota: "1r/h".parse().unwrap(),
        user_dht_burst: NonZeroU32::new(1),
        ..RateLimiterConfig::default()
    }))
    .await;
    let cached_owner = Keypair::random();
    relay.cache(&packet(&cached_owner, 300));
    let provider =
        RelayPacketProvider::new(&relay.service, Some(RealIp(Ipv4Addr::LOCALHOST.into())));
    provider
        .get_packet(&cached_owner.public_key())
        .await
        .unwrap();
    let missing = Keypair::random().public_key();
    for _ in 0..2 {
        let error = provider.get_packet(&missing).await.unwrap_err();
        assert_eq!(error.into_response().status(), StatusCode::NOT_FOUND);
    }
    let resolution = pkresolve::derive_resolution_async(&request(&cached_owner), &provider)
        .await
        .unwrap();
    assert_eq!(
        provider
            .into_bundle(resolution)
            .unwrap_err()
            .into_response()
            .status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn owner_limit_allows_repeated_keys_and_rejects_new_owners() {
    let relay = TestRelay::new(None).await;
    let provider = RelayPacketProvider::new(&relay.service, None);
    let first_owner = Keypair::random();
    relay.cache(&packet(&first_owner, 300));
    provider
        .get_packet(&first_owner.public_key())
        .await
        .unwrap();
    for _ in 1..MAX_RESOLUTION_PACKETS {
        let owner = Keypair::random();
        relay.cache(&packet(&owner, 300));
        provider.get_packet(&owner.public_key()).await.unwrap();
    }
    // The request retains this version even after eviction from the shared cache.
    assert!(relay
        .cache
        .get(&CacheKey::from(&first_owner.public_key()))
        .is_none());
    assert!(provider
        .get_packet(&first_owner.public_key())
        .await
        .unwrap()
        .is_some());
    let error = provider
        .get_packet(&Keypair::random().public_key())
        .await
        .unwrap_err();
    assert_eq!(
        error.into_response().status(),
        StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn alias_resolution_bundles_both_cached_packets_for_client_replay() {
    let relay = TestRelay::new(None).await;
    let origin = Keypair::random();
    let target = Keypair::random();
    let target_name = format!("_8443._https.app.{}", target.to_z32());
    let alias = SignedPacket::builder()
        .https(
            "_8443._https.app".try_into().unwrap(),
            SVCB::new(0, target_name.as_str().try_into().unwrap()),
            60,
        )
        .sign(&origin)
        .unwrap();
    relay.cache(&alias);
    relay.cache(&packet(&target, 300));
    let mut client = ServiceClient::default();
    let original = request(&origin);
    let Preparation::Fetch(fetch) = client.prepare(original.clone()).unwrap() else {
        panic!("empty cache");
    };
    let response = relay.post(fetch.body().to_owned()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 8192).await.unwrap();
    let bundle: ResolveBundle = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(bundle.packets.len(), 2);
    let resolution = client.accept_bundle(fetch, &bytes).unwrap();
    assert_eq!(&resolution.service, original.service());
    assert_eq!(resolution.candidates[0].port.get(), 8443);
    assert!(
        resolution
            .expires_at
            .saturating_duration_since(std::time::Instant::now())
            <= Duration::from_secs(60)
    );
    assert!(matches!(
        client.prepare(original).unwrap(),
        Preparation::Ready(_)
    ));
}
