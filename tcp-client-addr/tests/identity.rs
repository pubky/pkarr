use std::{net::SocketAddr, time::Duration};

use ipnet::IpNet;
use ppp::v2::{Addresses, Builder, Command, IPv4, IPv6, Protocol, Version};
use tcp_client_addr::{ConfigError, IdentifyError, IdentityMode, ProxyProtocol};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::timeout,
};

async fn connected_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (server, _) = listener.accept().await.unwrap();
    (client, server)
}

fn proxy_mode() -> IdentityMode {
    let loopback = "127.0.0.0/8".parse::<IpNet>().unwrap();
    IdentityMode::ProxyProtocol(ProxyProtocol::new([loopback]).unwrap())
}

#[tokio::test]
async fn direct_mode_keeps_the_entire_stream_and_uses_peer() {
    let (mut sender, stream) = connected_pair().await;
    let peer = stream.peer_addr().unwrap();
    sender
        .write_all(b"GET / HTTP/1.1\r\nX-Forwarded-For: 1.2.3.4\r\n\r\n")
        .await
        .unwrap();

    let (mut stream, addr) = IdentityMode::Direct.identify(stream).await.unwrap();
    assert_eq!(addr.client(), peer);
    assert_eq!(addr.peer(), peer);
    let mut first = [0; 3];
    stream.read_exact(&mut first).await.unwrap();
    assert_eq!(&first, b"GET");
}

#[tokio::test]
async fn v1_preface_consumes_no_application_bytes() {
    let (mut sender, stream) = connected_pair().await;
    let peer = stream.peer_addr().unwrap();
    sender
        .write_all(b"PROXY TCP4 198.51.100.8 192.0.2.1 4242 80\r\nhello")
        .await
        .unwrap();

    let (mut stream, addr) = proxy_mode().identify(stream).await.unwrap();
    assert_eq!(
        addr.client(),
        "198.51.100.8:4242".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(addr.peer(), peer);
    let mut body = [0; 5];
    stream.read_exact(&mut body).await.unwrap();
    assert_eq!(&body, b"hello");
}

#[tokio::test]
async fn v1_ipv6_source_is_identified() {
    let (mut sender, stream) = connected_pair().await;
    sender
        .write_all(b"PROXY TCP6 2001:db8::1 2001:db8::2 4242 80\r\n")
        .await
        .unwrap();

    let (_, addr) = proxy_mode().identify(stream).await.unwrap();
    assert_eq!(addr.client(), "[2001:db8::1]:4242".parse().unwrap());
}

#[tokio::test]
async fn v2_preface_consumes_no_application_bytes() {
    let (mut sender, stream) = connected_pair().await;
    let addresses: Addresses = IPv4::new([198, 51, 100, 9], [192, 0, 2, 1], 4243, 80).into();
    let mut bytes =
        Builder::with_addresses(Version::Two | Command::Proxy, Protocol::Stream, addresses)
            .write_tlv(0x01_u8, b"metadata")
            .unwrap()
            .build()
            .unwrap();
    bytes.extend_from_slice(b"hello");
    sender.write_all(&bytes).await.unwrap();

    let (mut stream, addr) = proxy_mode().identify(stream).await.unwrap();
    assert_eq!(
        addr.client(),
        "198.51.100.9:4243".parse::<SocketAddr>().unwrap()
    );
    let mut body = [0; 5];
    stream.read_exact(&mut body).await.unwrap();
    assert_eq!(&body, b"hello");
}

#[tokio::test]
async fn v2_ipv6_source_is_identified() {
    let (mut sender, stream) = connected_pair().await;
    let addresses: Addresses = IPv6::new(
        "2001:db8::1"
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets(),
        "2001:db8::2"
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets(),
        4242,
        80,
    )
    .into();
    let bytes = Builder::with_addresses(Version::Two | Command::Proxy, Protocol::Stream, addresses)
        .build()
        .unwrap();
    sender.write_all(&bytes).await.unwrap();

    let (_, addr) = proxy_mode().identify(stream).await.unwrap();
    assert_eq!(addr.client(), "[2001:db8::1]:4242".parse().unwrap());
}

// Independent wire fixture: v2 PROXY, TCP/IPv4, 12 address bytes,
// 198.51.100.9:4243 -> 192.0.2.1:80. Do not build this with the parser dependency.
const V2_TCP4: &[u8] =
    b"\r\n\r\n\0\r\nQUIT\n\x21\x11\x00\x0c\xc6\x33\x64\x09\xc0\x00\x02\x01\x10\x93\x00\x50";

#[tokio::test]
async fn fragmented_v2_preface_is_accepted() {
    let (mut sender, stream) = connected_pair().await;
    let mode = proxy_mode();
    let mut identifying = Box::pin(mode.identify(stream));

    // Poll the reader after every fragment so TCP coalescing cannot hide the
    // partial reads. Each fragment stops inside a different part of the header.
    for fragment in [&V2_TCP4[..5], &V2_TCP4[5..14], &V2_TCP4[14..20]] {
        sender.write_all(fragment).await.unwrap();
        tokio::select! {
            result = &mut identifying => panic!("accepted incomplete preface: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
    sender.write_all(&V2_TCP4[20..]).await.unwrap();
    sender.write_all(b"hello").await.unwrap();

    let (mut stream, addr) = identifying.await.unwrap();
    assert_eq!(addr.client(), "198.51.100.9:4243".parse().unwrap());
    let mut body = [0; 5];
    stream.read_exact(&mut body).await.unwrap();
    assert_eq!(&body, b"hello");
}

#[tokio::test]
async fn invalid_or_unsupported_v2_fixed_headers_are_rejected_without_payload() {
    for (version_command, family_protocol, payload_len, malformed) in [
        (0x31, 0x11, 12_u16, true), // invalid version
        (0x22, 0x11, 12, true),     // invalid command
        (0x21, 0x41, 12, true),     // invalid family
        (0x21, 0x13, 12, true),     // invalid transport
        (0x21, 0x11, 11, true),     // incomplete IPv4 address block
        (0x21, 0x21, 35, true),     // incomplete IPv6 address block
        (0x20, 0x00, 12, false),    // LOCAL with a payload
        (0x21, 0x12, 12, false),    // IPv4 UDP
        (0x21, 0x22, 36, false),    // IPv6 UDP
        (0x21, 0x31, 216, false),   // UNIX stream
        (0x21, 0x01, 12, false),    // unspecified family
        (0x21, 0x10, 12, false),    // unspecified transport
    ] {
        let (mut sender, stream) = connected_pair().await;
        let mut fixed = V2_TCP4[..16].to_vec();
        fixed[12] = version_command;
        fixed[13] = family_protocol;
        fixed[14..16].copy_from_slice(&payload_len.to_be_bytes());
        sender.write_all(&fixed).await.unwrap();

        // Keep the sender open and withhold its payload. The parser must decide
        // from the fixed header, not wait for EOF or its one-second timeout.
        let result = timeout(Duration::from_millis(500), proxy_mode().identify(stream))
            .await
            .expect("fixed header should be rejected before reading the payload");
        if malformed {
            assert!(matches!(result, Err(IdentifyError::Malformed)));
        } else {
            assert!(matches!(result, Err(IdentifyError::MissingTcpSource)));
        }
    }
}

#[tokio::test]
async fn v2_metadata_is_skipped_without_validation() {
    for metadata in [
        &b"\x01"[..],                         // incomplete TLV header
        &b"\x01\x00\x02x"[..],                // truncated TLV value
        &b"\x03\x00\x04\xff\xff\xff\xff"[..], // unchecked CRC32C
    ] {
        let (mut sender, stream) = connected_pair().await;
        let mut header = V2_TCP4.to_vec();
        let payload_len = (12 + metadata.len()) as u16;
        header[14..16].copy_from_slice(&payload_len.to_be_bytes());
        header.extend_from_slice(metadata);
        header.extend_from_slice(b"hello");
        sender.write_all(&header).await.unwrap();

        let (mut stream, addr) = proxy_mode().identify(stream).await.unwrap();
        assert_eq!(addr.client(), "198.51.100.9:4243".parse().unwrap());
        let mut body = [0; 5];
        stream.read_exact(&mut body).await.unwrap();
        assert_eq!(&body, b"hello");
    }
}

#[tokio::test]
async fn fragmented_v1_preface_is_accepted() {
    let (mut sender, stream) = connected_pair().await;
    let writer = tokio::spawn(async move {
        sender.write_all(b"PROXY TCP4 198.51.100.8").await.unwrap();
        tokio::task::yield_now().await;
        sender.write_all(b" 192.0.2.1 4242 80\r\nH").await.unwrap();
    });

    let (mut stream, addr) = proxy_mode().identify(stream).await.unwrap();
    writer.await.unwrap();
    assert_eq!(
        addr.client(),
        "198.51.100.8:4242".parse::<SocketAddr>().unwrap()
    );
    let mut byte = [0];
    stream.read_exact(&mut byte).await.unwrap();
    assert_eq!(&byte, b"H");
}

#[tokio::test]
async fn plain_http_and_unknown_v1_do_not_fall_back_to_peer() {
    let (mut sender, stream) = connected_pair().await;
    sender.write_all(b"GET / HTTP/1.1\r\n").await.unwrap();
    assert!(matches!(
        proxy_mode().identify(stream).await,
        Err(IdentifyError::Malformed)
    ));

    let (mut sender, stream) = connected_pair().await;
    sender.write_all(b"PROXY UNKNOWN\r\n").await.unwrap();
    assert!(matches!(
        proxy_mode().identify(stream).await,
        Err(IdentifyError::MissingTcpSource)
    ));
}

#[tokio::test]
async fn untrusted_peer_is_rejected_before_reading() {
    let (_, stream) = connected_pair().await;
    let config = ProxyProtocol::new(["192.0.2.0/24".parse::<IpNet>().unwrap()]).unwrap();
    let result = IdentityMode::ProxyProtocol(config).identify(stream).await;
    assert!(matches!(result, Err(IdentifyError::UntrustedPeer(_))));
}

#[tokio::test]
async fn oversized_and_truncated_headers_are_rejected() {
    let (mut sender, stream) = connected_pair().await;
    sender
        .write_all(b"PROXY TCP4 198.51.100.8 192.0.2.1 4242 80\r\n")
        .await
        .unwrap();
    let config = ProxyProtocol::new(["127.0.0.0/8".parse::<IpNet>().unwrap()])
        .unwrap()
        .with_max_header_bytes(28)
        .unwrap();
    assert!(matches!(
        IdentityMode::ProxyProtocol(config).identify(stream).await,
        Err(IdentifyError::HeaderTooLong)
    ));

    let (mut sender, stream) = connected_pair().await;
    sender.write_all(b"PROXY TCP4 198.51.100.8").await.unwrap();
    drop(sender);
    assert!(matches!(
        proxy_mode().identify(stream).await,
        Err(IdentifyError::UnexpectedEof)
    ));
}

#[tokio::test]
async fn maximum_length_v1_tcp6_header_obeys_configured_limit() {
    let source = "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff";
    let header = format!("PROXY TCP6 {source} {source} 65535 65535\r\n");
    assert_eq!(header.len(), 104);

    for (limit, accepted) in [(104, true), (103, false)] {
        let (mut sender, stream) = connected_pair().await;
        sender.write_all(header.as_bytes()).await.unwrap();
        let config = ProxyProtocol::new(["127.0.0.1/32".parse().unwrap()])
            .unwrap()
            .with_max_header_bytes(limit)
            .unwrap();
        let result = IdentityMode::ProxyProtocol(config).identify(stream).await;

        if accepted {
            assert_eq!(
                result.unwrap().1.client_ip(),
                source.parse::<std::net::IpAddr>().unwrap()
            );
        } else {
            assert!(matches!(result, Err(IdentifyError::HeaderTooLong)));
        }
    }
}

#[tokio::test]
async fn v1_header_over_protocol_limit_is_rejected_without_waiting_for_eof() {
    let (mut sender, stream) = connected_pair().await;
    let mut bytes = b"PROXY ".to_vec();
    bytes.resize(107, b'a');
    sender.write_all(&bytes).await.unwrap();

    let result = timeout(Duration::from_millis(500), proxy_mode().identify(stream))
        .await
        .expect("overlong v1 header should be rejected immediately");
    assert!(matches!(result, Err(IdentifyError::HeaderTooLong)));
}

#[tokio::test]
async fn oversized_v2_header_is_rejected_without_waiting_for_its_payload() {
    let (mut sender, stream) = connected_pair().await;
    let mut fixed = b"\r\n\r\n\0\r\nQUIT\n".to_vec();
    fixed.extend_from_slice(&[0x21, 0x11, 0x10, 0x00]);
    sender.write_all(&fixed).await.unwrap();

    let result = timeout(Duration::from_secs(1), proxy_mode().identify(stream))
        .await
        .unwrap();
    assert!(matches!(result, Err(IdentifyError::HeaderTooLong)));
}

#[tokio::test]
async fn v2_header_obeys_configured_limit_including_tlvs() {
    let addresses: Addresses = IPv4::new([198, 51, 100, 9], [192, 0, 2, 1], 4243, 80).into();
    let header =
        Builder::with_addresses(Version::Two | Command::Proxy, Protocol::Stream, addresses)
            .write_tlv(0x01_u8, b"")
            .unwrap()
            .build()
            .unwrap();
    assert_eq!(header.len(), 31);

    for (limit, accepted) in [(31, true), (30, false)] {
        let (mut sender, stream) = connected_pair().await;
        sender.write_all(&header).await.unwrap();
        let config = ProxyProtocol::new(["127.0.0.1/32".parse().unwrap()])
            .unwrap()
            .with_max_header_bytes(limit)
            .unwrap();
        let result = IdentityMode::ProxyProtocol(config).identify(stream).await;

        if accepted {
            assert_eq!(
                result.unwrap().1.client_ip(),
                "198.51.100.9".parse::<std::net::IpAddr>().unwrap()
            );
        } else {
            assert!(matches!(result, Err(IdentifyError::HeaderTooLong)));
        }
    }
}

#[tokio::test]
async fn mutated_protocol_signatures_are_rejected() {
    let v1 = b"PROXY TCP4 198.51.100.8 192.0.2.1 4242 80\r\n";
    for index in 0..6 {
        let (mut sender, stream) = connected_pair().await;
        let mut header = v1.to_vec();
        header[index] ^= 1;
        sender.write_all(&header).await.unwrap();
        drop(sender);
        assert!(matches!(
            proxy_mode().identify(stream).await,
            Err(IdentifyError::Malformed)
        ));
    }

    let addresses: Addresses = IPv4::new([198, 51, 100, 9], [192, 0, 2, 1], 4243, 80).into();
    let v2 = Builder::with_addresses(Version::Two | Command::Proxy, Protocol::Stream, addresses)
        .build()
        .unwrap();
    for index in 0..12 {
        let (mut sender, stream) = connected_pair().await;
        let mut header = v2.clone();
        header[index] ^= 1;
        sender.write_all(&header).await.unwrap();
        drop(sender);
        assert!(matches!(
            proxy_mode().identify(stream).await,
            Err(IdentifyError::Malformed)
        ));
    }
}

#[tokio::test]
async fn every_truncated_prefix_of_a_valid_v1_or_v2_header_is_rejected() {
    let v1 = b"PROXY TCP4 198.51.100.8 192.0.2.1 4242 80\r\n".to_vec();
    let addresses: Addresses = IPv4::new([198, 51, 100, 9], [192, 0, 2, 1], 4243, 80).into();
    let v2 = Builder::with_addresses(Version::Two | Command::Proxy, Protocol::Stream, addresses)
        .build()
        .unwrap();

    for header in [&v1, &v2] {
        for end in 0..header.len() {
            let (mut sender, stream) = connected_pair().await;
            sender.write_all(&header[..end]).await.unwrap();
            drop(sender);
            assert!(
                proxy_mode().identify(stream).await.is_err(),
                "accepted a header truncated at byte {end}"
            );
        }
    }
}

#[tokio::test]
async fn v2_local_command_and_truncated_payload_are_rejected() {
    let (mut sender, stream) = connected_pair().await;
    let mut local = b"\r\n\r\n\0\r\nQUIT\n".to_vec();
    local.extend_from_slice(&[0x20, 0x00, 0x00, 0x00]);
    sender.write_all(&local).await.unwrap();
    assert!(matches!(
        proxy_mode().identify(stream).await,
        Err(IdentifyError::MissingTcpSource)
    ));

    let (mut sender, stream) = connected_pair().await;
    let mut truncated = b"\r\n\r\n\0\r\nQUIT\n".to_vec();
    truncated.extend_from_slice(&[0x21, 0x11, 0x00, 0x0c]);
    sender.write_all(&truncated).await.unwrap();
    drop(sender);
    assert!(matches!(
        proxy_mode().identify(stream).await,
        Err(IdentifyError::UnexpectedEof)
    ));
}

#[tokio::test]
async fn incomplete_preface_times_out_by_default() {
    let (mut sender, stream) = connected_pair().await;
    let result = timeout(Duration::from_secs(2), proxy_mode().identify(stream))
        .await
        .expect("built-in timeout should finish first");
    assert!(matches!(result, Err(IdentifyError::HeaderTimeout)));
    assert_connection_closed(&mut sender).await;
}

async fn assert_connection_closed(sender: &mut TcpStream) {
    let mut byte = [0];
    let result = timeout(Duration::from_secs(1), sender.read(&mut byte))
        .await
        .expect("identified connection should have been closed");
    assert!(matches!(result, Ok(0)));
}

#[tokio::test]
async fn trickled_bytes_do_not_restart_the_header_timeout() {
    let (mut sender, stream) = connected_pair().await;
    sender.write_all(b"PROXY ").await.unwrap();
    let writer = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(30)).await;
            if sender.write_all(b"T").await.is_err() {
                break;
            }
        }
    });
    let config = ProxyProtocol::new(["127.0.0.1/32".parse().unwrap()])
        .unwrap()
        .with_header_timeout(Duration::from_millis(150))
        .unwrap();
    let result = timeout(
        Duration::from_millis(600),
        IdentityMode::ProxyProtocol(config).identify(stream),
    )
    .await;
    writer.abort();
    let _ = writer.await;
    assert!(matches!(result, Ok(Err(IdentifyError::HeaderTimeout))));
}

#[tokio::test]
async fn configured_timeout_covers_the_complete_preface() {
    let (mut sender, stream) = connected_pair().await;
    sender.write_all(b"PROXY ").await.unwrap();
    let config = ProxyProtocol::new(["127.0.0.0/8".parse::<IpNet>().unwrap()])
        .unwrap()
        .with_header_timeout(Duration::from_millis(20))
        .unwrap();

    let result = timeout(
        Duration::from_secs(1),
        IdentityMode::ProxyProtocol(config).identify(stream),
    )
    .await
    .expect("configured timeout should finish first");
    assert!(matches!(result, Err(IdentifyError::HeaderTimeout)));
}

#[tokio::test]
async fn caller_can_still_apply_a_stricter_acceptance_deadline() {
    let (mut sender, stream) = connected_pair().await;
    assert!(
        timeout(Duration::from_millis(20), proxy_mode().identify(stream))
            .await
            .is_err()
    );
    assert_connection_closed(&mut sender).await;
}

#[test]
fn configuration_requires_trusted_peers_and_a_bounded_header() {
    assert!(matches!(
        ProxyProtocol::new([]),
        Err(ConfigError::NoTrustedProxies)
    ));
    let config = ProxyProtocol::new(["127.0.0.0/8".parse::<IpNet>().unwrap()]).unwrap();
    assert!(matches!(
        config.clone().with_header_timeout(Duration::ZERO),
        Err(ConfigError::InvalidHeaderTimeout)
    ));
    assert!(matches!(
        config.clone().with_max_header_bytes(27),
        Err(ConfigError::InvalidHeaderLimit)
    ));
    assert!(matches!(
        config.with_max_header_bytes(65_552),
        Err(ConfigError::InvalidHeaderLimit)
    ));
}
