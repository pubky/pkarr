//! Provider traversal, resource bounds, failure propagation, and cancellation.

use std::{
    collections::HashMap,
    future::pending,
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime},
};

use futures_lite::future::{block_on, poll_once};
use pkarr::{dns::rdata::SVCB, Keypair, SignedPacket, SignedPacketBuilder};
use pkresolve::{
    derive_resolution, derive_resolution_async, AsyncPacketProvider, Error, HttpProtocol,
    PacketProvider, PublicKey, ResolutionRequest,
};

fn now() -> SystemTime {
    SystemTime::now()
}
fn key(seed: u8) -> Keypair {
    Keypair::from_secret_key(&[seed; 32])
}
fn proof(builder: SignedPacketBuilder, key: &Keypair) -> SignedPacket {
    let mut packet = builder.timestamp(now().into()).sign(key).unwrap();
    packet.set_last_seen(&now().into());
    packet
}
fn request(key: &Keypair) -> ResolutionRequest {
    ResolutionRequest::new(
        key.to_z32().parse().unwrap(),
        443,
        vec![HttpProtocol::Http11],
    )
    .unwrap()
}
fn packet_cache(packets: Vec<SignedPacket>) -> HashMap<PublicKey, Arc<SignedPacket>> {
    packets
        .into_iter()
        .map(|packet| (packet.public_key(), Arc::new(packet)))
        .collect()
}
fn delegated(alice: &Keypair, provider: &Keypair) -> Vec<SignedPacket> {
    let target = format!("_443._tcp.{}", provider.to_z32());
    let alternate = format!("_8443._tcp.{}", provider.to_z32());
    let mut svc = SVCB::new(2, ".".try_into().unwrap());
    svc.set_port(8443);
    vec![
        proof(
            SignedPacket::builder()
                .https(
                    ".".try_into().unwrap(),
                    SVCB::new(1, ".".try_into().unwrap()),
                    300,
                )
                .https(".".try_into().unwrap(), svc, 300)
                .address(
                    ".".try_into().unwrap(),
                    "203.0.113.10".parse().unwrap(),
                    300,
                )
                .cname(
                    "_443._tcp".try_into().unwrap(),
                    target.as_str().try_into().unwrap(),
                    300,
                )
                .cname(
                    "_8443._tcp".try_into().unwrap(),
                    alternate.as_str().try_into().unwrap(),
                    300,
                ),
            alice,
        ),
        proof(SignedPacket::builder(), provider),
    ]
}

enum Reply {
    Packet(Arc<SignedPacket>),
    Failed,
    Pending,
}
struct TestProvider {
    replies: HashMap<PublicKey, Reply>,
    requests: Mutex<Vec<PublicKey>>,
    cancelled: AtomicBool,
}
impl TestProvider {
    fn new(proofs: &[SignedPacket]) -> Self {
        Self {
            replies: proofs
                .iter()
                .map(|proof| (proof.public_key(), Reply::Packet(Arc::new(proof.clone()))))
                .collect(),
            requests: Mutex::new(Vec::new()),
            cancelled: AtomicBool::new(false),
        }
    }
    fn requested(&self) -> Vec<PublicKey> {
        self.requests.lock().unwrap().clone()
    }
}
struct DropFlag<'a>(&'a AtomicBool);
impl Drop for DropFlag<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
impl AsyncPacketProvider for TestProvider {
    type Error = io::Error;
    async fn get_packet(&self, key: &PublicKey) -> Result<Option<Arc<SignedPacket>>, io::Error> {
        self.requests.lock().unwrap().push(key.clone());
        match self.replies.get(key) {
            Some(Reply::Packet(proof)) => {
                futures_lite::future::yield_now().await;
                Ok(Some(proof.clone()))
            }
            Some(Reply::Failed) => Err(io::Error::new(io::ErrorKind::TimedOut, "source timed out")),
            Some(Reply::Pending) => {
                let _guard = DropFlag(&self.cancelled);
                pending().await
            }
            None => Ok(None),
        }
    }
}

#[test]
fn resolution_only_queries_the_destination_packet() {
    let alice = key(1);
    let provider = key(2);
    let proofs = delegated(&alice, &provider);
    let source = TestProvider::new(&proofs);
    let result = block_on(derive_resolution_async(&request(&alice), &source)).unwrap();
    assert_eq!(source.requested(), vec![alice.public_key()]);
    assert_eq!(result.candidates.len(), 2);
    assert_eq!(result.service.public_key(), &alice.public_key());
    assert_eq!(&result.service, request(&alice).service());
}

#[test]
fn response_bundle_uses_the_same_resolution_path_without_network_fallback() {
    let alice = key(1);
    let provider = key(2);
    let proofs = delegated(&alice, &provider);
    let bundle = packet_cache(proofs.clone());
    let result = derive_resolution(&request(&alice), &bundle).unwrap();
    assert_eq!(result.candidates.len(), 2);
    // Delegation targets are not required for direct destination-key TLS.
    let destination_only = packet_cache(proofs[..1].to_vec());
    assert!(derive_resolution(&request(&alice), &destination_only).is_ok());
}

#[test]
fn missing_destination_packet_is_an_error() {
    let alice = key(1);
    let source = TestProvider::new(&[]);
    assert!(matches!(
        block_on(derive_resolution_async(&request(&alice), &source)),
        Err(Error::MissingPacket(_))
    ));
    assert_eq!(source.requested().len(), 1);
}

#[test]
fn resolver_does_not_repeat_provider_packet_validation() {
    let alice = key(1);
    let mut proofs = delegated(&alice, &key(2));
    let mut bytes = proofs[0].serialize();
    bytes[8 + 40] ^= 1;
    proofs[0] = SignedPacket::deserialize(&bytes).unwrap();
    // This deliberately violates the provider contract to confirm that the
    // resolver interprets supplied packets rather than rechecking signatures.
    let source = TestProvider::new(&proofs);
    assert!(block_on(derive_resolution_async(&request(&alice), &source)).is_ok());
}

#[test]
fn provider_error_keeps_its_underlying_failure() {
    let alice = key(1);
    let mut source = TestProvider::new(&[]);
    source.replies.insert(alice.public_key(), Reply::Failed);
    let error = block_on(derive_resolution_async(&request(&alice), &source)).unwrap_err();
    let Error::Provider { key, source } = error else {
        panic!("expected provider error");
    };
    assert_eq!(*key, alice.public_key());
    assert_eq!(
        source.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::TimedOut
    );
}

#[test]
fn synchronous_provider_needs_no_sync_bound_and_preserves_errors() {
    struct FailingProvider {
        requests: std::cell::RefCell<Vec<PublicKey>>,
    }
    impl PacketProvider for FailingProvider {
        type Error = io::Error;

        fn get_packet(&self, key: &PublicKey) -> Result<Option<Arc<SignedPacket>>, io::Error> {
            self.requests.borrow_mut().push(key.clone());
            Err(io::Error::new(io::ErrorKind::TimedOut, "source timed out"))
        }
    }

    let alice = key(1);
    let provider = FailingProvider {
        requests: Default::default(),
    };
    let source: &dyn PacketProvider<Error = io::Error> = &provider;
    let error = derive_resolution(&request(&alice), source).unwrap_err();
    let Error::Provider { key, source } = error else {
        panic!("expected provider error");
    };
    assert_eq!(*key, alice.public_key());
    assert_eq!(provider.requests.into_inner(), vec![alice.public_key()]);
    assert_eq!(
        source.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::TimedOut
    );
}

#[test]
fn async_lookup_can_suspend_then_return_the_same_routes_as_sync_lookup() {
    let alice = key(1);
    let proofs = delegated(&alice, &key(2));
    let cache = packet_cache(proofs.clone());
    let provider = TestProvider::new(&proofs);
    let req = request(&alice);
    let sync_result = derive_resolution(&req, &cache).unwrap();

    let mut future = Box::pin(derive_resolution_async(&req, &provider));
    assert!(block_on(poll_once(future.as_mut())).is_none());
    let async_result = block_on(future).unwrap();

    assert_eq!(async_result.service, sync_result.service);
    assert_eq!(async_result.url_port, sync_result.url_port);
    assert_eq!(async_result.candidates, sync_result.candidates);
}

#[test]
fn time_waiting_for_the_provider_counts_against_record_ttl() {
    use std::{convert::Infallible, future::poll_fn, task::Poll};

    struct DelayedProvider {
        packet: Arc<SignedPacket>,
        ready: AtomicBool,
    }
    impl AsyncPacketProvider for DelayedProvider {
        type Error = Infallible;
        async fn get_packet(&self, _: &PublicKey) -> Result<Option<Arc<SignedPacket>>, Infallible> {
            poll_fn(|_| {
                if self.ready.load(Ordering::SeqCst) {
                    Poll::Ready(Ok(Some(self.packet.clone())))
                } else {
                    Poll::Pending
                }
            })
            .await
        }
    }
    let alice = key(1);
    let packet = proof(
        SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(1, ".".try_into().unwrap()),
                300,
            )
            .address(".".try_into().unwrap(), "203.0.113.10".parse().unwrap(), 1),
        &alice,
    );
    let source = DelayedProvider {
        packet: Arc::new(packet),
        ready: AtomicBool::new(false),
    };
    let req = request(&alice);
    let mut future = Box::pin(derive_resolution_async(&req, &source));
    assert!(block_on(poll_once(future.as_mut())).is_none());
    std::thread::sleep(Duration::from_millis(1100));
    source.ready.store(true, Ordering::SeqCst);
    assert!(matches!(block_on(future), Err(Error::ExpiredRecord)));
}

#[test]
fn dropping_resolution_drops_pending_lookup_without_further_queries() {
    let alice = key(1);
    let mut source = TestProvider::new(&[]);
    source.replies.insert(alice.public_key(), Reply::Pending);
    let req = request(&alice);
    let mut future = Box::pin(derive_resolution_async(&req, &source));
    fn requires_send<T: Send>(_: &T) {}
    requires_send(&future);
    assert!(block_on(poll_once(future.as_mut())).is_none());
    assert_eq!(source.requested(), vec![alice.public_key()]);
    assert!(!source.cancelled.load(Ordering::SeqCst));
    drop(future);
    assert!(source.cancelled.load(Ordering::SeqCst));
    assert_eq!(source.requested().len(), 1);
}

#[test]
fn cache_clones_and_bundle_lookups_share_the_decoded_packet() {
    let alice = key(1);
    let proofs = delegated(&alice, &key(2));
    let bundle = packet_cache(proofs.clone());
    let first = bundle.get_packet(&alice.public_key()).unwrap().unwrap();
    let second = bundle.get_packet(&alice.public_key()).unwrap().unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    let cloned = first.clone();
    assert!(Arc::ptr_eq(&first, &cloned));
    assert_eq!(first.last_seen(), cloned.last_seen());
}

#[test]
fn local_cache_miss_can_be_filled_from_a_server_bundle_before_retrying() {
    let alice = key(1);
    let provider = key(2);
    let proofs = delegated(&alice, &provider);
    let mut cache = HashMap::new();
    assert!(matches!(derive_resolution(&request(&alice), &cache),
        Err(Error::MissingPacket(missing)) if *missing == alice.public_key()));
    // Represents the HTTP adapter obtaining a complete /v1/resolve response.
    let response = packet_cache(proofs.clone());
    for packet in response.values() {
        cache.insert(packet.public_key(), packet.clone());
    }
    let result = derive_resolution(&request(&alice), &cache).unwrap();
    assert_eq!(result.candidates.len(), 2);
    let key = provider.public_key();
    assert!(Arc::ptr_eq(&cache[&key], &response[&key]));
}

#[test]
fn expired_cached_packets_are_rejected_and_can_be_replaced_from_server() {
    let alice = key(1);
    let proofs = delegated(&alice, &key(2));
    let mut expired = proofs[0].clone();
    expired.set_last_seen(&(now() - Duration::from_secs(601)).into());
    let mut cache = HashMap::from([(alice.public_key(), Arc::new(expired))]);
    assert!(matches!(
        derive_resolution(&request(&alice), &cache),
        Err(Error::ExpiredRecord)
    ));
    let response = packet_cache(proofs.clone());
    for packet in response.values() {
        cache.insert(packet.public_key(), packet.clone());
    }
    assert!(derive_resolution(&request(&alice), &cache).is_ok());
    // TTL expiry is also checked on an otherwise cryptographically valid cache hit.
    let mut expired = proofs[0].clone();
    expired.set_last_seen(&(now() - Duration::from_secs(301)).into());
    cache.insert(alice.public_key(), Arc::new(expired));
    assert!(matches!(
        derive_resolution(&request(&alice), &cache),
        Err(Error::ExpiredRecord)
    ));
}

#[test]
fn unsupported_protocols_are_filtered_without_tls_lookups() {
    use pkarr::dns::rdata::SVCParam;

    let alice = key(1);
    let mut http3 = SVCB::new(1, ".".try_into().unwrap()).with_param(SVCParam::NoDefaultAlpn);
    http3.set_alpn(&["h3".try_into().unwrap()]);
    http3.set_port(8443);
    let missing_delegate = format!("_8443._tcp.{}", key(2).to_z32());
    let packet = proof(
        SignedPacket::builder()
            .https(".".try_into().unwrap(), http3, 300)
            .https(
                ".".try_into().unwrap(),
                SVCB::new(2, ".".try_into().unwrap()),
                300,
            )
            .address(
                ".".try_into().unwrap(),
                "203.0.113.10".parse().unwrap(),
                300,
            )
            .cname(
                "_8443._tcp".try_into().unwrap(),
                missing_delegate.as_str().try_into().unwrap(),
                300,
            ),
        &alice,
    );
    let source = TestProvider::new(&[packet]);
    let result = block_on(derive_resolution_async(&request(&alice), &source)).unwrap();
    assert_eq!(source.requested(), vec![alice.public_key()]);
    let candidates = &result.candidates;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].port.get(), 443);
    assert_eq!(candidates[0].protocols.as_slice(), &[HttpProtocol::Http11]);
}
