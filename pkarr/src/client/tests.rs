//! Client native tests

use std::{thread, time::Duration};

use pkarr_relay::Relay;
use pubky_timestamp::Timestamp;
use rstest::rstest;

use crate::errors::{BuildError, ConcurrencyError, PublishError};
use crate::{Client, ClientBuilder, Keypair, SignedPacket};

#[derive(Copy, Clone)]
enum Networks {
    Dht,
    #[cfg(feature = "relays")]
    Relays,
    Both,
}

/// Parametric [ClientBuilder] with no default networks,
/// instead it uses mainline or relays depending on `networks` enum.
fn builder(relay: &Relay, testnet: &mainline::Testnet, networks: Networks) -> ClientBuilder {
    let mut builder = Client::builder();

    builder
        .no_default_network()
        // Because of pkarr_relay crate, dht is always enabled.
        .bootstrap(&testnet.bootstrap)
        .request_timeout(Duration::from_millis(100));

    match networks {
        Networks::Dht => {}
        #[cfg(feature = "relays")]
        Networks::Relays => {
            builder
                .no_default_network()
                .relays(&[relay.local_url()])
                .unwrap();
        }
        Networks::Both => {
            #[cfg(feature = "relays")]
            {
                builder.relays(&[relay.local_url()]).unwrap();
            }
        }
    }

    builder
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn publish_resolve(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet, None).await.unwrap();

    let b = builder(&relay, &testnet, networks).build().unwrap();

    let resolved = b.resolve(&keypair.public_key()).await.unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

    let from_cache = b.resolve(&keypair.public_key()).await.unwrap();
    assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
    assert_eq!(from_cache.last_seen(), resolved.last_seen());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn client_send(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet, None).await.unwrap();

    let b = builder(&relay, &testnet, networks).build().unwrap();

    tokio::spawn(async move {
        let resolved = b.resolve(&keypair.public_key()).await.unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    })
    .await
    .unwrap();
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn return_expired_packet_fallback(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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

    let resolved = client.resolve(&keypair.public_key()).await;

    assert_eq!(resolved, Some(signed_packet));
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn ttl_0_test(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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

    client.publish(&signed_packet, None).await.unwrap();

    // First Call
    let resolved = client.resolve(&signed_packet.public_key()).await.unwrap();

    assert_eq!(resolved.encoded_packet(), signed_packet.encoded_packet());

    thread::sleep(Duration::from_millis(10));

    let second = client.resolve(&signed_packet.public_key()).await.unwrap();
    assert_eq!(second.encoded_packet(), signed_packet.encoded_packet());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn not_found(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let resolved = client.resolve(&keypair.public_key()).await;

    assert_eq!(resolved, None);
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
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn repeated_publish_query(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    client.publish(&signed_packet, None).await.unwrap();

    client.publish(&signed_packet, None).await.unwrap()
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_resolve(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();
    let b = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet, None).await.unwrap();

    let public_key = signed_packet.public_key();
    let bclone = b.clone();
    let _stream = tokio::spawn(async move { bclone.resolve(&public_key).await.unwrap() });

    let response_second = b.resolve(&signed_packet.public_key()).await.unwrap();

    assert_eq!(&response_second.as_bytes(), &signed_packet.as_bytes());

    assert!(_stream.await.is_ok())
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_publish_same_packet(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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
            client.publish(&signed_packet, None).await.unwrap()
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_publish_of_different_packets(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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
            let result = client.publish(&signed_packet, None).await;

            if i == 0 {
                result.unwrap();
            } else {
                assert!(matches!(
                    result,
                    Err(PublishError::Concurrency(ConcurrencyError::ConflictRisk))
                ))
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
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn concurrent_publish_different_with_cas(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let cloned_client = client.clone();
    let cloned_keypair = keypair.clone();
    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&cloned_keypair)
        .unwrap();

    let cas = Some(signed_packet.timestamp());

    let handle = tokio::spawn(async move {
        assert!(matches!(
            cloned_client.publish(&signed_packet, None).await,
            Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent))
        ));
    });

    // Second
    {
        let signed_packet = SignedPacket::builder()
            .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
            .sign(&keypair)
            .unwrap();

        client.publish(&signed_packet, cas).await.unwrap();
    }

    handle.await.unwrap();
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn conflict_302(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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
            None,
        )
        .await
        .unwrap();

    assert!(matches!(
        client
            .publish(
                &signed_packet_builder.timestamp(t1).sign(&keypair).unwrap(),
                None
            )
            .await,
        Err(PublishError::Concurrency(ConcurrencyError::NotMostRecent))
    ));
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn conflict_301_cas(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
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
            None,
        )
        .await
        .unwrap();

    assert!(matches!(
        client
            .publish(&signed_packet_builder.sign(&keypair).unwrap(), Some(t1))
            .await,
        Err(PublishError::Concurrency(ConcurrencyError::CasFailed))
    ));
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[test]
fn blocking(#[case] networks: Networks) {
    let (relay, testnet) = futures_lite::future::block_on(async_compat::Compat::new(async {
        let testnet = mainline::Testnet::new(10).unwrap();
        let relay = Relay::run_test(&testnet).await.unwrap();

        (relay, testnet)
    }));

    let a = builder(&relay, &testnet, networks)
        .build()
        .unwrap()
        .as_blocking();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder()
        .txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30)
        .sign(&keypair)
        .unwrap();

    a.publish(&signed_packet, None).unwrap();

    let b = builder(&relay, &testnet, networks)
        .build()
        .unwrap()
        .as_blocking();

    let resolved = b.resolve(&keypair.public_key()).unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

    let from_cache = b.resolve(&keypair.public_key()).unwrap();
    assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
    assert_eq!(from_cache.last_seen(), resolved.last_seen());
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[test]
fn no_tokio(#[case] networks: Networks) {
    futures_lite::future::block_on(async {
        let (relay, testnet) = async_compat::Compat::new(async {
            let testnet = mainline::Testnet::new(10).unwrap();
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

        a.publish(&signed_packet, None).await.unwrap();

        let b = builder(&relay, &testnet, networks).build().unwrap();

        let resolved = b.resolve(&keypair.public_key()).await.unwrap();
        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());

        let from_cache = b.resolve(&keypair.public_key()).await.unwrap();
        assert_eq!(from_cache.as_bytes(), signed_packet.as_bytes());
        assert_eq!(from_cache.last_seen(), resolved.last_seen());
    });
}

#[rstest]
#[case::dht(Networks::Dht)]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn zero_cache_size(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let a = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet = SignedPacket::builder().sign(&keypair).unwrap();

    a.publish(&signed_packet, None).await.unwrap();

    let b = builder(&relay, &testnet, networks)
        .cache_size(0)
        .build()
        .unwrap();

    let resolved = b.resolve(&keypair.public_key()).await.unwrap();
    assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
}

#[rstest]
#[case::both_networks(Networks::Both)]
#[cfg_attr(feature = "relays", case::relays(Networks::Relays))]
#[tokio::test]
async fn clear_inflight_requests(#[case] networks: Networks) {
    let testnet = mainline::Testnet::new(10).unwrap();
    let relay = Relay::run_test(&testnet).await.unwrap();

    let client = builder(&relay, &testnet, networks).build().unwrap();

    let keypair = Keypair::random();

    let signed_packet_builder =
        SignedPacket::builder().txt("foo".try_into().unwrap(), "bar".try_into().unwrap(), 30);

    client
        .publish(&signed_packet_builder.clone().sign(&keypair).unwrap(), None)
        .await
        .unwrap();

    // If there was a memory leak, we would get a `ConflictRisk` error instead.
    client
        .publish(&signed_packet_builder.sign(&keypair).unwrap(), None)
        .await
        .unwrap();
}
