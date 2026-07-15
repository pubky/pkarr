//! Client native tests

use std::net::Ipv4Addr;
use std::{thread, time::Duration};

use ntimestamp::Timestamp;
use pkarr_relay::Relay;
use rstest::rstest;
use simple_dns::rdata::SVCB;

use crate::errors::{BuildError, PublishError, ResolveError};
use crate::{Client, ClientBuilder, Keypair, ResolvePolicy, SignedPacket};

#[derive(Copy, Clone)]
pub(crate) enum Networks {
    Dht,
    #[cfg(feature = "relays")]
    Relays,
    Combined,
}

/// Parameterized [`ClientBuilder`] with no default networks.
///
/// Configures mainline, relays, or both according to `networks`.
pub(crate) fn builder(
    _relay: &Relay,
    testnet: &mainline::Testnet,
    networks: Networks,
) -> ClientBuilder {
    let mut builder = Client::builder();

    builder
        .no_default_network()
        // Because of pkarr_relay crate, dht is always enabled.
        .bootstrap(&testnet.bootstrap)
        .dht(|config| {
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        });

    if std::env::var("CI").is_ok() {
        builder.request_timeout(Duration::from_millis(1000));
    } else {
        builder.request_timeout(Duration::from_millis(500));
    }

    match networks {
        Networks::Dht => {}
        #[cfg(feature = "relays")]
        Networks::Relays => {
            builder
                .no_default_network()
                .relays(&[_relay.local_url()])
                .unwrap();
        }
        Networks::Combined => {
            #[cfg(feature = "relays")]
            {
                builder.relays(&[_relay.local_url()]).unwrap();
            }
        }
    }

    builder
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn publish_resolve(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet).await.unwrap();

    let b = builder(&relay, &testnet, networks).build().unwrap();

    let resolved = b
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

    let from_cache = b
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();
    assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
    assert_eq!(from_cache.last_seen(), resolved.last_seen());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn client_send(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet).await.unwrap();

    let b = builder(&relay, &testnet, networks).build().unwrap();

    tokio::spawn(async move {
        let resolved = b
            .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
            .await
            .unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b
            .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
            .await
            .unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    })
    .await
    .unwrap();
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn return_expired_packet_fallback(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks)
        .maximum_ttl(0)
        .build()
        .unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    client
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &signed_packet);

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap_err();
    assert_eq!(resolved, ResolveError::NotFound);

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheOnly)
        .await
        .unwrap();
    assert_eq!(resolved, signed_packet);
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn cache_first_rejects_network_packet_older_than_expired_cache(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let publisher = builder(&relay, &testnet, networks).build().unwrap();
    let resolver = builder(&relay, &testnet, networks)
        .maximum_ttl(0)
        .build()
        .unwrap();

    let keypair = Keypair::random();
    let signed_packet_builder =
        SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);
    let older = signed_packet_builder
        .clone()
        .timestamp(Timestamp::from(1))
        .sign(&keypair)
        .unwrap();
    let newer = signed_packet_builder
        .timestamp(Timestamp::from(2))
        .sign(&keypair)
        .unwrap();

    publisher.publish(&older).await.unwrap();
    resolver
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &newer);

    let resolved = resolver
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap_err();
    assert_eq!(resolved, ResolveError::NotFound);

    let local = resolver
        .resolve(&keypair.public_key(), ResolvePolicy::CacheOnly)
        .await
        .unwrap();
    assert_eq!(local, newer);
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn network_only_ignores_newer_local_cache(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let publisher = builder(&relay, &testnet, networks).build().unwrap();
    let resolver = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet_builder =
        SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);
    let older = signed_packet_builder
        .clone()
        .timestamp(Timestamp::from(1))
        .sign(&keypair)
        .unwrap();
    let newer = signed_packet_builder
        .timestamp(Timestamp::from(2))
        .sign(&keypair)
        .unwrap();

    publisher.publish(&older).await.unwrap();
    resolver
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &newer);

    let resolved = resolver
        .resolve(&keypair.public_key(), ResolvePolicy::NetworkOnly)
        .await
        .unwrap();

    assert_eq!(resolved.as_bytes(), older.as_bytes());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn ttl_0_test(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks)
        .maximum_ttl(0)
        .build()
        .unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        // Include sub-second precision to exercise relay cache bounds.
        .timestamp(Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap() + 500_000)
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    client.publish(&signed_packet).await.unwrap();

    // First Call
    let resolved = client
        .resolve(&signed_packet.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();

    assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

    thread::sleep(Duration::from_millis(10));

    let second = client
        .resolve(&signed_packet.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();
    assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn not_found(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap_err();

    assert_eq!(resolved, ResolveError::NotFound);
}

#[test]
fn no_network() {
    assert!(matches!(
        Client::builder().no_default_network().build(),
        Err(BuildError::NoNetwork)
    ));
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn repeated_publish_query(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    client.publish(&signed_packet).await.unwrap();

    client.publish(&signed_packet).await.unwrap();
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn relay_publish_waits_past_first_error() {
    let (failed_relay, successful_relay) = spawn_ordered_publish_relays(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        axum::http::StatusCode::NO_CONTENT,
    );

    let client = Client::builder()
        .no_default_network()
        .request_timeout(Duration::from_secs(1))
        .relays(&[failed_relay, successful_relay])
        .unwrap()
        .build()
        .unwrap();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&Keypair::random())
        .unwrap();

    client.publish(&signed_packet).await.unwrap();
}

#[cfg(feature = "relays")]
#[derive(Clone, Default)]
struct RelayResponseGate {
    wait_for: Option<std::sync::Arc<tokio::sync::Notify>>,
    signal_next: Option<std::sync::Arc<tokio::sync::Notify>>,
}

#[cfg(feature = "relays")]
impl RelayResponseGate {
    async fn wait(&self) {
        if let Some(wait_for) = &self.wait_for {
            wait_for.notified().await;
        }
    }

    fn signal_next(&self) {
        if let Some(signal_next) = &self.signal_next {
            signal_next.notify_one();
        }
    }
}

#[cfg(feature = "relays")]
fn ordered_response_gates() -> (RelayResponseGate, RelayResponseGate) {
    let first_response_ready = std::sync::Arc::new(tokio::sync::Notify::new());

    (
        RelayResponseGate {
            signal_next: Some(first_response_ready.clone()),
            ..RelayResponseGate::default()
        },
        RelayResponseGate {
            wait_for: Some(first_response_ready),
            ..RelayResponseGate::default()
        },
    )
}

#[cfg(feature = "relays")]
#[derive(Clone)]
struct PacketRelayState {
    packet: std::sync::Arc<SignedPacket>,
    memento_datetime: Option<Timestamp>,
    response_gate: RelayResponseGate,
}

#[cfg(feature = "relays")]
async fn packet_relay_handler(
    axum::extract::State(state): axum::extract::State<PacketRelayState>,
) -> axum::response::Response {
    state.response_gate.wait().await;

    let mut response = axum::response::Response::builder();
    if let Some(timestamp) = state.memento_datetime {
        response = response.header("memento-datetime", timestamp.format_http_date());
    }

    let response = response
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(state.packet.to_relay_payload()))
        .unwrap();
    state.response_gate.signal_next();
    response
}

#[cfg(feature = "relays")]
fn spawn_packet_relay(packet: SignedPacket) -> url::Url {
    spawn_controlled_packet_relay(packet, None, RelayResponseGate::default())
}

#[cfg(feature = "relays")]
fn spawn_packet_relay_with_memento(
    packet: SignedPacket,
    memento_datetime: Option<Timestamp>,
) -> url::Url {
    spawn_controlled_packet_relay(packet, memento_datetime, RelayResponseGate::default())
}

#[cfg(feature = "relays")]
fn spawn_ordered_packet_relays(
    first: (SignedPacket, Option<Timestamp>),
    second: (SignedPacket, Option<Timestamp>),
) -> (url::Url, url::Url) {
    let (first_gate, second_gate) = ordered_response_gates();

    (
        spawn_controlled_packet_relay(first.0, first.1, first_gate),
        spawn_controlled_packet_relay(second.0, second.1, second_gate),
    )
}

#[cfg(feature = "relays")]
fn spawn_controlled_packet_relay(
    packet: SignedPacket,
    memento_datetime: Option<Timestamp>,
    response_gate: RelayResponseGate,
) -> url::Url {
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();

    let app = axum::Router::new()
        .route("/{key}", axum::routing::get(packet_relay_handler))
        .with_state(PacketRelayState {
            packet: std::sync::Arc::new(packet),
            memento_datetime,
            response_gate,
        });

    tokio::spawn(
        axum_server::from_tcp(listener)
            .unwrap()
            .serve(app.into_make_service()),
    );

    format!("http://{address}").parse().unwrap()
}

#[cfg(feature = "relays")]
fn spawn_publish_relay(status: axum::http::StatusCode) -> url::Url {
    spawn_controlled_publish_relay(status, RelayResponseGate::default())
}

#[cfg(feature = "relays")]
fn spawn_ordered_publish_relays(
    first: axum::http::StatusCode,
    second: axum::http::StatusCode,
) -> (url::Url, url::Url) {
    let (first_gate, second_gate) = ordered_response_gates();

    (
        spawn_controlled_publish_relay(first, first_gate),
        spawn_controlled_publish_relay(second, second_gate),
    )
}

#[cfg(feature = "relays")]
fn spawn_controlled_publish_relay(
    status: axum::http::StatusCode,
    response_gate: RelayResponseGate,
) -> url::Url {
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();

    let app = axum::Router::new().route(
        "/{key}",
        axum::routing::put(move || {
            let response_gate = response_gate.clone();

            async move {
                response_gate.wait().await;
                let response = axum::response::Response::builder()
                    .status(status)
                    .body(axum::body::Body::empty())
                    .unwrap();
                response_gate.signal_next();
                response
            }
        }),
    );

    tokio::spawn(
        axum_server::from_tcp(listener)
            .unwrap()
            .serve(app.into_make_service()),
    );

    format!("http://{address}").parse().unwrap()
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn cache_only_relay_result_is_cached() {
    let keypair = Keypair::random();
    let packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();
    let relay = spawn_packet_relay(packet.clone());
    let client = Client::builder()
        .no_default_network()
        .request_timeout(Duration::from_secs(1))
        .relays(&[relay])
        .unwrap()
        .build()
        .unwrap();

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheOnly)
        .await
        .unwrap();

    assert_eq!(resolved.as_bytes(), packet.as_bytes());
    let cached = client
        .cache()
        .unwrap()
        .get_read_only(&keypair.public_key().into())
        .unwrap();
    assert_eq!(cached.as_bytes(), packet.as_bytes());
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn relay_most_recent_resolve_aggregates_all_relays() {
    let keypair = Keypair::random();

    let older = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "old".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(1))
        .sign(&keypair)
        .unwrap();
    let newer = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "new".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(2))
        .sign(&keypair)
        .unwrap();

    let (older_relay, newer_relay) =
        spawn_ordered_packet_relays((older, None), (newer.clone(), None));

    let client = Client::builder()
        .no_default_network()
        .request_timeout(Duration::from_secs(1))
        .relays(&[older_relay, newer_relay])
        .unwrap()
        .build()
        .unwrap();

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::NetworkOnly)
        .await
        .unwrap();

    assert_eq!(resolved.timestamp(), newer.timestamp());
    assert_eq!(resolved.as_bytes(), newer.as_bytes());
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn cache_first_waits_for_relay_packet_above_cache_floor() {
    let keypair = Keypair::random();
    let first = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "first".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(10))
        .sign(&keypair)
        .unwrap();
    let second = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "second".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(10))
        .sign(&keypair)
        .unwrap();
    let (mut cached, below_floor) = if first.more_recent_than(&second) {
        (first, second)
    } else {
        (second, first)
    };
    cached.set_last_seen(&(Timestamp::now() - 60 * 1_000_000_u64));
    let above_floor = SignedPacket::builder()
        .timestamp(Timestamp::from(11))
        .sign(&keypair)
        .unwrap();

    let (below_floor_relay, above_floor_relay) =
        spawn_ordered_packet_relays((below_floor, None), (above_floor.clone(), None));
    let client = Client::builder()
        .no_default_network()
        .maximum_ttl(30)
        .request_timeout(Duration::from_secs(1))
        .relays(&[below_floor_relay, above_floor_relay])
        .unwrap()
        .build()
        .unwrap();
    client
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &cached);

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();

    assert_eq!(resolved.timestamp(), above_floor.timestamp());
    assert_eq!(resolved.as_bytes(), above_floor.as_bytes());
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn cache_first_caches_newer_expired_relay_packet() {
    let keypair = Keypair::random();
    let mut cached = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "cached".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(10))
        .sign(&keypair)
        .unwrap();
    let packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "newer".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(11))
        .sign(&keypair)
        .unwrap();
    let last_seen = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
    cached.set_last_seen(&(last_seen - 60 * 1_000_000_u64));
    let relay = spawn_packet_relay_with_memento(packet.clone(), Some(last_seen));
    let client = Client::builder()
        .no_default_network()
        .maximum_ttl(30)
        .request_timeout(Duration::from_secs(1))
        .relays(&[relay])
        .unwrap()
        .build()
        .unwrap();
    client
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &cached);

    let result = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await;

    assert_eq!(result, Err(ResolveError::NotFound));
    let cached = client
        .cache()
        .unwrap()
        .get_read_only(&keypair.public_key().into())
        .unwrap();
    assert_eq!(cached.as_bytes(), packet.as_bytes());
    assert_eq!(cached.last_seen(), &last_seen);
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn cache_first_waits_for_fresh_relay_after_expired_response() {
    let keypair = Keypair::random();
    let mut cached = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "cached".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(9))
        .sign(&keypair)
        .unwrap();
    let expired = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "expired".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(11))
        .sign(&keypair)
        .unwrap();
    let fresh = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "fresh".try_into().unwrap(), 30)
        .timestamp(Timestamp::from(10))
        .sign(&keypair)
        .unwrap();
    let expired_last_seen = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
    cached.set_last_seen(&(expired_last_seen - 60 * 1_000_000_u64));
    let (expired_relay, fresh_relay) =
        spawn_ordered_packet_relays((expired, Some(expired_last_seen)), (fresh.clone(), None));
    let client = Client::builder()
        .no_default_network()
        .maximum_ttl(30)
        .request_timeout(Duration::from_secs(1))
        .relays(&[expired_relay, fresh_relay])
        .unwrap()
        .build()
        .unwrap();
    client
        .cache()
        .unwrap()
        .put(&keypair.public_key().into(), &cached);

    let resolved = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();

    assert_eq!(resolved.as_bytes(), fresh.as_bytes());
    assert!(!resolved.is_expired(0, 30));
    let cached = client
        .cache()
        .unwrap()
        .get_read_only(&keypair.public_key().into())
        .unwrap();
    assert_eq!(cached.as_bytes(), fresh.as_bytes());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_resolve(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();
    let b = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet).await.unwrap();

    let public_key = signed_packet.public_key();
    let bclone = b.clone();
    let _stream = tokio::spawn(async move {
        bclone
            .resolve(&public_key, ResolvePolicy::CacheFirst)
            .await
            .unwrap()
    });

    let response_second = b
        .resolve(&signed_packet.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();

    assert_eq!(&response_second.as_bytes(), &signed_packet.as_bytes());

    assert!(_stream.await.is_ok())
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_publish_same_packet(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    let mut handles = vec![];

    for _ in 0..2 {
        let client = client.clone();
        let signed_packet = signed_packet.clone();

        handles.push(tokio::spawn(async move {
            client.publish(&signed_packet).await.unwrap();
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_publish_of_different_packets(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let mut handles = vec![];

    let keypair = Keypair::random();

    let timestamp = Timestamp::now();

    for i in 0..1 {
        let client = client.clone();

        let signed_packet = SignedPacket::builder()
            .txt(
                format!("foo{i}").as_str().try_into().unwrap(),
                "bar".try_into().unwrap(),
                30,
            )
            .timestamp(timestamp)
            .sign(&keypair)
            .unwrap();

        handles.push(tokio::spawn(async move {
            let result = client.publish(&signed_packet).await;

            if i == 0 {
                result.unwrap();
            } else {
                assert!(matches!(result, Err(PublishError::NotMostRecent)))
            }
        }));

        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn conflict_302(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(10).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet_builder =
        SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);

    let t1 = Timestamp::now();
    let t2 = Timestamp::now();

    client
        .publish(
            &signed_packet_builder
                .clone()
                .timestamp(t2)
                .sign(&keypair)
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(matches!(
        client
            .publish(&signed_packet_builder.timestamp(t1).sign(&keypair).unwrap())
            .await,
        Err(PublishError::NotMostRecent)
    ));
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn relay_publish_rejects_conflicting_relay_quorum() {
    let success = spawn_publish_relay(axum::http::StatusCode::NO_CONTENT);
    let conflict1 = spawn_publish_relay(axum::http::StatusCode::CONFLICT);
    let conflict2 = spawn_publish_relay(axum::http::StatusCode::CONFLICT);

    let client = Client::builder()
        .no_default_network()
        .request_timeout(Duration::from_secs(1))
        .relays(&[success, conflict1, conflict2])
        .unwrap()
        .build()
        .unwrap();

    let signed_packet = SignedPacket::builder().sign(&Keypair::random()).unwrap();

    assert!(matches!(
        client.publish(&signed_packet).await,
        Err(PublishError::NotMostRecent)
    ));
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn combined_publish_does_not_mask_relay_concurrency_error() {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = spawn_publish_relay(axum::http::StatusCode::CONFLICT);

    let client = Client::builder()
        .no_default_network()
        .bootstrap(&testnet.bootstrap)
        .dht(|config| {
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        })
        .request_timeout(Duration::from_millis(200))
        .relays(&[relay])
        .unwrap()
        .build()
        .unwrap();

    let signed_packet = SignedPacket::builder().sign(&Keypair::random()).unwrap();

    assert!(matches!(
        client.publish(&signed_packet).await,
        Err(PublishError::NotMostRecent)
    ));
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn combined_publish_does_not_mask_dht_not_most_recent_after_relay_success() {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = spawn_publish_relay(axum::http::StatusCode::NO_CONTENT);

    let dht_client = Client::builder()
        .no_default_network()
        .bootstrap(&testnet.bootstrap)
        .dht(|config| {
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        })
        .request_timeout(Duration::from_millis(200))
        .build()
        .unwrap();

    let client = Client::builder()
        .no_default_network()
        .bootstrap(&testnet.bootstrap)
        .dht(|config| {
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        })
        .request_timeout(Duration::from_millis(200))
        .relays(&[relay])
        .unwrap()
        .build()
        .unwrap();

    let keypair = Keypair::random();
    let signed_packet_builder =
        SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);
    let older = signed_packet_builder
        .clone()
        .timestamp(Timestamp::from(1))
        .sign(&keypair)
        .unwrap();
    let newer = signed_packet_builder
        .timestamp(Timestamp::from(2))
        .sign(&keypair)
        .unwrap();

    dht_client.publish(&newer).await.unwrap();

    assert!(matches!(
        client.publish(&older).await,
        Err(PublishError::NotMostRecent)
    ));
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[test]
fn no_tokio(#[case] networks: Networks) {
    futures_lite::future::block_on(async {
        let (relay, testnet) = async_compat::Compat::new(async {
            let testnet = mainline::Testnet::builder(5).build().unwrap();
            let relay = Relay::run_test(&testnet).await.unwrap();

            (relay, testnet)
        })
        .await;

        let a = builder(&relay, &testnet, networks).build().unwrap();

        let keypair = Keypair::random();

        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        a.publish(&signed_packet).await.unwrap();

        let b = builder(&relay, &testnet, networks).build().unwrap();

        let resolved = b
            .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
            .await
            .unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b
            .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
            .await
            .unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    });
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn zero_cache_size(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

    a.publish(&signed_packet).await.unwrap();

    let b = builder(&relay, &testnet, networks)
        .cache_size(0)
        .build()
        .unwrap();

    let resolved = b
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
}

// #[rstest]
// #[case::combined_networks(Networks::Combined)]
// #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
// #[tokio::test]
// async fn clear_inflight_requests(#[case] networks: Networks) {
//     let testnet = mainline::Testnet::builder(5).build().unwrap();
//     let relay = Relay::run_test(&testnet).await.unwrap();
//
//     let client = builder(&relay, &testnet, networks).build().unwrap();
//
//     let keypair = Keypair::random();
//
//     let signed_packet_builder =
//         SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);
//
//     client
//         .publish(&signed_packet_builder.clone().sign(&keypair).unwrap())
//         .await
//         .unwrap();
//
//     tokio::time::sleep(Duration::from_millis(200)).await;
//
//     // If there was a memory leak, publish would fail instead.
//     client
//         .publish(&signed_packet_builder.sign(&keypair).unwrap())
//         .await
//         .unwrap();
// }

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn publish_resolve_most_recent_with_no_cache(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet).await.unwrap();

    let b = builder(&relay, &testnet, networks)
        .cache_size(0)
        .build()
        .unwrap();

    let resolved = b
        .resolve(&keypair.public_key(), ResolvePolicy::NetworkOnly)
        .await
        .unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::combined_networks(Networks::Combined)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn republish_after_resolve(#[case] networks: Networks) {
    let testnet = mainline::Testnet::builder(5).build().unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .cname("test".try_into().unwrap(), "test2".try_into().unwrap(), 600)
        .sign(&keypair)
        .unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    client.publish(&signed_packet).await.unwrap();

    client
        .resolve(&keypair.public_key(), ResolvePolicy::NetworkOnly)
        .await
        .expect("valid signed packet");

    let new_packet = SignedPacket::builder()
        .cname("test".try_into().unwrap(), "test2".try_into().unwrap(), 600)
        .sign(&keypair)
        .unwrap();

    client.publish(&new_packet).await.unwrap();
}

#[cfg(feature = "relays")]
#[tokio::test]
async fn discard_cache_with_zero_capacity() {
    use std::net::ToSocketAddrs;

    let testnet = crate::mainline::Testnet::builder(2).build().unwrap();

    // Create relay
    let storage = std::env::temp_dir().join(Timestamp::now().to_string());
    let mut builder = pkarr_relay::Relay::builder();
    builder
        .disable_rate_limiter()
        .cache_size(0) // Comment this line out and it works
        .http_port(0)
        .storage(storage)
        .dht(|config| {
            config.bootstrap = Some(
                testnet
                    .bootstrap
                    .iter()
                    .filter_map(|address| address.to_socket_addrs().ok())
                    .flatten()
                    .filter_map(|address| match address {
                        std::net::SocketAddr::V4(address) => Some(address),
                        std::net::SocketAddr::V6(_) => None,
                    })
                    .collect(),
            );
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        });
    let relay = unsafe { builder.run().await.unwrap() };
    let relay_url = relay.local_url();

    let keypair = Keypair::random();

    // Publish packet on the DHT without using the relay.
    let client = Client::builder()
        .no_default_network()
        .bootstrap(&testnet.bootstrap)
        .dht(|config| {
            config.bind_address = Some(Ipv4Addr::LOCALHOST);
            config
        })
        .build()
        .unwrap();

    let signed_packet = SignedPacket::builder()
        .txt(
            "example.com".try_into().unwrap(),
            "foo".try_into().unwrap(),
            300,
        )
        .sign(&keypair)
        .unwrap();
    client.publish(&signed_packet).await.unwrap();

    // Resolve packet with only the relay, no DHT
    let client = Client::builder()
        .no_default_network()
        .relays(&[relay_url])
        .unwrap()
        .build()
        .unwrap();
    let packet = client
        .resolve(&keypair.public_key(), ResolvePolicy::CacheFirst)
        .await
        .unwrap();
    assert_eq!(packet.public_key(), keypair.public_key());
}

#[tokio::test]
async fn regression_relay_timeout_stack_overflow() {
    let host: crate::dns::Name = "example.com".try_into().unwrap();
    let svcb = SVCB::new(0, host);
    let signed_packet_builder =
        SignedPacket::builder().https("_pubky".try_into().unwrap(), svcb.clone(), 60 * 60);

    let client = Client::builder()
        .no_dht()
        .request_timeout(Duration::from_millis(100))
        .build()
        .unwrap();

    let kp = Keypair::random();
    let pkt = signed_packet_builder.clone().sign(&kp).unwrap();

    let _ = client.publish(&pkt).await;
}

#[cfg(feature = "reqwest-builder")]
mod reqwest_builder {
    use super::*;

    use std::{
        net::{SocketAddr, TcpListener},
        sync::Arc,
    };

    use axum::routing::get;
    use axum::Router;
    use axum_server::tls_rustls::RustlsConfig;

    use crate::{dns::rdata::SVCB, Client, Keypair, SignedPacket};

    async fn publish_server_pkarr(client: &Client, keypair: &Keypair, socket_addr: &SocketAddr) {
        let mut svcb = SVCB::new(0, ".".try_into().unwrap());
        svcb.set_port(socket_addr.port());

        let signed_packet = SignedPacket::builder()
            .https(".".try_into().unwrap(), svcb, 60 * 60)
            .address(".".try_into().unwrap(), socket_addr.ip(), 60 * 60)
            .sign(keypair)
            .unwrap();

        client.publish(&signed_packet).await.unwrap();
    }

    #[rstest]
    #[case::dht(Networks::Dht)]
    #[case::combined_networks(Networks::Combined)]
    #[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
    #[tokio::test]
    async fn reqwest_pkarr_domain(#[case] networks: Networks) {
        let testnet = mainline::Testnet::builder(5).build().unwrap();
        let relay = Relay::run_test(&testnet).await.unwrap();

        let keypair = Keypair::random();

        {
            // Run a server on Pkarr
            let app = Router::new().route("/", get(|| async { "Hello, world!" }));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap(); // Bind to any available port
            listener.set_nonblocking(true).unwrap();
            let address = listener.local_addr().unwrap();

            let client = builder(&relay, &testnet, networks).build().unwrap();
            publish_server_pkarr(&client, &keypair, &address).await;

            println!("Server running on https://{}", keypair.public_key());

            let server = axum_server::from_tcp_rustls(
                listener,
                RustlsConfig::from_config(Arc::new((&keypair).into())),
            )
            .unwrap();

            tokio::spawn(server.serve(app.into_make_service()));
        }

        // Client setup
        let pkarr_client = builder(&relay, &testnet, networks).build().unwrap();
        let reqwest = reqwest::ClientBuilder::from(pkarr_client).build().unwrap();

        // Make a request
        let response = reqwest
            .get(format!("https://{}", keypair.public_key()))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), "Hello, world!");
    }
}
