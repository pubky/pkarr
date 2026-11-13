use std::{
    io,
    net::{Ipv4Addr, SocketAddr, TcpListener},
    time::Duration,
};

use axum::Router;
use h2::Ping;
use http::StatusCode;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::{sleep, timeout},
};

use super::{HttpServer, Limits, HTTP1_MAX_BUFFER_SIZE};

#[tokio::test]
async fn connection_without_request_hits_initial_request_header_timeout() {
    let server = TestServer::spawn(Limits {
        initial_request_header_timeout: Duration::from_millis(200),
        max_connection_age: Duration::from_secs(1),
        ..test_limits()
    });
    let mut stream = TcpStream::connect(server.address).await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("connection should close at its initial request header deadline")
        .unwrap();
}

#[tokio::test]
async fn partial_protocol_preface_hits_initial_request_header_timeout() {
    let server = TestServer::spawn(Limits {
        initial_request_header_timeout: Duration::from_millis(200),
        max_connection_age: Duration::from_secs(1),
        ..test_limits()
    });
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream.write_all(b"P").await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("partial protocol detection should reach the initial request header deadline")
        .unwrap();
}

#[tokio::test]
async fn incomplete_http2_handshake_hits_initial_request_header_timeout() {
    let server = TestServer::spawn(Limits {
        initial_request_header_timeout: Duration::from_millis(200),
        max_connection_age: Duration::from_secs(1),
        ..test_limits()
    });
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream
        .write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
        .await
        .unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("an incomplete HTTP/2 handshake should reach the initial request header deadline")
        .unwrap();
}

#[tokio::test]
async fn complete_request_headers_disarm_initial_request_header_timeout() {
    let server = TestServer::spawn(Limits {
        initial_request_header_timeout: Duration::from_millis(200),
        max_connection_age: Duration::from_secs(1),
        ..test_limits()
    });
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);

    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();
    assert_eq!(response.await.unwrap().status(), StatusCode::NOT_FOUND);

    sleep(Duration::from_millis(300)).await;
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();
    assert_eq!(response.await.unwrap().status(), StatusCode::NOT_FOUND);

    server.server.shutdown();
    timeout(Duration::from_secs(1), connection_task)
        .await
        .expect("connection should close on shutdown")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn idle_connection_is_closed_at_maximum_age() {
    let server = TestServer::spawn(Limits {
        max_connection_age: Duration::from_millis(200),
        ..test_limits()
    });
    let mut stream = TcpStream::connect(server.address).await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("idle connection should reach its maximum age")
        .unwrap();
}

#[tokio::test]
async fn incomplete_http1_headers_hit_header_timeout() {
    let server = TestServer::spawn(Limits {
        max_connection_age: Duration::from_secs(1),
        http1_header_read_timeout: Duration::from_millis(200),
        ..test_limits()
    });
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream.write_all(b"GET / HTTP/1.1\r\nHost:").await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("incomplete headers should time out")
        .unwrap();
}

#[tokio::test]
async fn oversized_http1_headers_are_rejected() {
    let server = TestServer::spawn(test_limits());
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    let oversized_header = "a".repeat(HTTP1_MAX_BUFFER_SIZE);
    let request =
        format!("GET / HTTP/1.1\r\nHost: localhost\r\nX-Large: {oversized_header}\r\n\r\n");

    stream.write_all(request.as_bytes()).await.unwrap();

    let mut response = Vec::new();
    timeout(Duration::from_secs(1), stream.read_to_end(&mut response))
        .await
        .expect("oversized HTTP/1 headers should be rejected promptly")
        .unwrap();

    assert!(response.starts_with(b"HTTP/1.1 431"));
}

#[tokio::test]
async fn active_http2_stream_finishes_during_grace_period() {
    let app = Router::new().route(
        "/",
        axum::routing::get(|| async {
            sleep(Duration::from_millis(300)).await;
            StatusCode::NO_CONTENT
        }),
    );
    let server = TestServer::spawn_with_app(
        Limits {
            max_connection_age: Duration::from_millis(200),
            drain_timeout: Duration::from_millis(500),
            ..test_limits()
        },
        app,
    );
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();

    let response = timeout(Duration::from_secs(1), response)
        .await
        .expect("request should finish during the grace period")
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    timeout(Duration::from_secs(1), connection_task)
        .await
        .expect("connection should close after draining")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn http2_stream_exceeding_grace_period_is_interrupted() {
    let app = Router::new().route(
        "/",
        axum::routing::get(|| async {
            sleep(Duration::from_secs(1)).await;
            StatusCode::NO_CONTENT
        }),
    );
    let server = TestServer::spawn_with_app(
        Limits {
            max_connection_age: Duration::from_millis(200),
            drain_timeout: Duration::from_millis(200),
            ..test_limits()
        },
        app,
    );
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();

    assert!(timeout(Duration::from_secs(1), response)
        .await
        .expect("connection should close after the grace period")
        .is_err());

    timeout(Duration::from_secs(1), connection_task)
        .await
        .expect("client connection task should finish")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn http2_ping_traffic_does_not_extend_maximum_age() {
    let server = TestServer::spawn(Limits {
        max_connection_age: Duration::from_millis(200),
        ..test_limits()
    });
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (_client, mut connection) = h2::client::handshake(stream).await.unwrap();
    let mut ping_pong = connection.ping_pong().unwrap();
    let connection_task = tokio::spawn(connection);
    let ping_task = tokio::spawn(async move {
        loop {
            sleep(Duration::from_millis(50)).await;
            if ping_pong.ping(Ping::opaque()).await.is_err() {
                break;
            }
        }
    });

    timeout(Duration::from_secs(1), connection_task)
        .await
        .expect("PING traffic should not prevent graceful shutdown")
        .unwrap()
        .unwrap();

    ping_task.abort();
}

#[tokio::test]
async fn excess_connection_is_refused() {
    let server = TestServer::spawn(Limits {
        max_connections: 1,
        max_connection_age: Duration::from_secs(1),
        ..test_limits()
    });
    let _first = TcpStream::connect(server.address).await.unwrap();
    sleep(Duration::from_millis(100)).await;
    let mut excess = TcpStream::connect(server.address).await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut excess))
        .await
        .expect("connection above the limit should be refused")
        .unwrap();
}

#[tokio::test]
async fn shutdown_closes_existing_connections() {
    let server = TestServer::spawn(test_limits());
    let mut stream = TcpStream::connect(server.address).await.unwrap();

    server.server.shutdown();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("server shutdown should close existing connections")
        .unwrap();
}

#[tokio::test]
async fn shutdown_interrupts_connection_drain() {
    let app = Router::new().route(
        "/",
        axum::routing::get(|| async {
            sleep(Duration::from_secs(5)).await;
            StatusCode::NO_CONTENT
        }),
    );
    let server = TestServer::spawn_with_app(
        Limits {
            max_connection_age: Duration::from_millis(200),
            drain_timeout: Duration::from_secs(2),
            ..test_limits()
        },
        app,
    );
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (_response, _) = client.send_request(request, true).unwrap();

    sleep(Duration::from_millis(300)).await;
    server.server.shutdown();

    timeout(Duration::from_millis(500), connection_task)
        .await
        .expect("shutdown should interrupt the connection drain")
        .unwrap()
        .unwrap();
}

fn test_limits() -> Limits {
    Limits {
        max_connections: 16,
        initial_request_header_timeout: Duration::from_secs(1),
        max_connection_age: Duration::from_secs(1),
        drain_timeout: Duration::from_millis(100),
        http1_header_read_timeout: Duration::from_secs(1),
    }
}

struct TestServer {
    address: SocketAddr,
    server: HttpServer,
}

impl TestServer {
    fn spawn(limits: Limits) -> Self {
        Self::spawn_with_app(limits, Router::new())
    }

    fn spawn_with_app(limits: Limits, app: Router) -> Self {
        Self::spawn_with_identity(limits, app, crate::ClientIdentityConfig::default())
    }

    fn spawn_with_identity(
        limits: Limits,
        app: Router,
        identity: crate::ClientIdentityConfig,
    ) -> Self {
        let identity = crate::real_ip::ClientIdentity::from_config(identity).unwrap();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = HttpServer::spawn(listener, app, limits, identity).unwrap();

        Self { address, server }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.server.shutdown();
    }
}

async fn wait_for_disconnect(stream: &mut TcpStream) -> io::Result<()> {
    let mut buffer = [0; 1024];

    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::NotConnected
                ) =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        }
    }
}

fn header_identity(network: &str) -> crate::ClientIdentityConfig {
    crate::ClientIdentityConfig::HttpHeader {
        header: "X-Real-IP".into(),
        trusted_proxies: vec![network.parse().unwrap()],
    }
}

fn identity_app() -> Router {
    Router::new().route(
        "/",
        axum::routing::get(
            |axum::Extension(ip): axum::Extension<crate::real_ip::RealIp>| async move {
                ip.0.to_string()
            },
        ),
    )
}

#[tokio::test]
async fn untrusted_proxy_is_disconnected_before_sending_http() {
    let server = TestServer::spawn_with_identity(
        Limits {
            initial_request_header_timeout: Duration::from_secs(30),
            max_connection_age: Duration::from_secs(30),
            ..test_limits()
        },
        identity_app(),
        header_identity("192.0.2.0/24"),
    );
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("untrusted peer should be rejected without waiting for HTTP bytes")
        .unwrap();
}

#[tokio::test]
async fn direct_identity_ignores_forwarding_headers() {
    let server = TestServer::spawn_with_app(test_limits(), identity_app());
    let response = reqwest::Client::new()
        .get(format!("http://{}/", server.address))
        .header("x-real-ip", "198.51.100.1")
        .header("x-forwarded-for", "198.51.100.2")
        .header("forwarded", "for=198.51.100.3")
        .header("cf-connecting-ip", "198.51.100.4")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), "127.0.0.1");
}

#[tokio::test]
async fn header_identity_is_required_on_all_routes_without_rate_limits() {
    let server = TestServer::spawn_with_identity(
        test_limits(),
        identity_app(),
        header_identity("127.0.0.1/32"),
    );
    let client = reqwest::Client::new();
    for path in ["/", "/missing"] {
        for values in [
            vec![],
            vec![""],
            vec!["garbage"],
            vec!["é"],
            vec!["198.51.100.1:80"],
            vec!["[2001:db8::1]:80"],
            vec!["[2001:db8::1]"],
            vec!["198.51.100.1, 198.51.100.2"],
            vec!["198.51.100.1", "198.51.100.1"],
        ] {
            let mut request = client
                .get(format!("http://{}{path}", server.address))
                .header("x-forwarded-for", "198.51.100.1")
                .header("cf-connecting-ip", "198.51.100.1");
            for value in &values {
                request = request.header("x-real-ip", *value);
            }
            let response = request.send().await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{path}: {values:?}"
            );
            let _ = response.bytes().await.unwrap();
        }
    }
    for (ip, canonical_ip) in [
        ("198.51.100.1", "198.51.100.1"),
        ("2001:db8::1", "2001:db8::1"),
        ("::ffff:198.51.100.1", "198.51.100.1"),
    ] {
        let response = client
            .get(format!("http://{}/", server.address))
            .header("x-real-ip", ip)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), canonical_ip);
    }
}

#[tokio::test]
async fn header_identity_is_required_before_cors_preflights() {
    let server = TestServer::spawn_with_identity(
        test_limits(),
        identity_app().layer(crate::cors_layer()),
        header_identity("127.0.0.1/32"),
    );
    let client = reqwest::Client::new();
    for (ip, expected) in [
        (None, StatusCode::BAD_REQUEST),
        (Some("invalid"), StatusCode::BAD_REQUEST),
        (Some("198.51.100.1"), StatusCode::OK),
    ] {
        let mut request = client
            .request(http::Method::OPTIONS, format!("http://{}/", server.address))
            .header("origin", "https://example.com")
            .header("access-control-request-method", "GET");
        if let Some(ip) = ip {
            request = request.header("x-real-ip", ip);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), expected);
        if expected == StatusCode::OK {
            assert_eq!(
                response.headers()["access-control-allow-origin"],
                "https://example.com"
            );
        }
    }
}

#[tokio::test]
async fn header_identity_is_resolved_for_each_http1_request_before_rate_limiting() {
    let config = crate::RateLimiterConfig {
        quota: "1r/h".parse().unwrap(),
        ..Default::default()
    };
    let limiter = crate::rate_limiting::IpRateLimiter::new(&config)
        .await
        .unwrap();
    let server = TestServer::spawn_with_identity(
        test_limits(),
        limiter.layer(identity_app()),
        header_identity("127.0.0.1/32"),
    );
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    // One proxy connection carries several clients, including a mapped IPv4 alias.
    let mut requests = String::new();
    for header in [
        "X-Real-IP: 198.51.100.1\r\n",
        "",
        "X-Real-IP: invalid\r\n",
        "X-Real-IP: ::ffff:198.51.100.1\r\n",
        "X-Real-IP: 2001:db8::1\r\n",
    ] {
        requests.push_str(&format!(
            "GET / HTTP/1.1\r\nHost: localhost\r\n{header}\r\n"
        ));
    }
    requests.push_str(
        "GET / HTTP/1.1\r\nHost: localhost\r\nX-Real-IP: 2001:db8::1\r\nConnection: close\r\n\r\n",
    );
    stream.write_all(requests.as_bytes()).await.unwrap();
    let mut response = String::new();
    timeout(Duration::from_secs(1), stream.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    let statuses: Vec<_> = response
        .split("HTTP/1.1 ")
        .skip(1)
        .map(|line| line.split_whitespace().next().unwrap())
        .collect();
    assert_eq!(
        statuses,
        ["200", "400", "400", "429", "200", "429"],
        "{response}"
    );
}

#[tokio::test]
async fn header_identity_is_resolved_for_each_http2_stream() {
    let config = crate::RateLimiterConfig {
        quota: "1r/h".parse().unwrap(),
        ..Default::default()
    };
    let limiter = crate::rate_limiting::IpRateLimiter::new(&config)
        .await
        .unwrap();
    let server = TestServer::spawn_with_identity(
        test_limits(),
        limiter.layer(identity_app()),
        header_identity("127.0.0.1/32"),
    );
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    for (ip, expected) in [
        (Some("198.51.100.1"), StatusCode::OK),
        (None, StatusCode::BAD_REQUEST),
        (Some("invalid"), StatusCode::BAD_REQUEST),
        (Some("::ffff:198.51.100.1"), StatusCode::TOO_MANY_REQUESTS),
        (Some("2001:db8::1"), StatusCode::OK),
        (Some("2001:db8::1"), StatusCode::TOO_MANY_REQUESTS),
    ] {
        let mut request = http::Request::get(format!("http://{}/", server.address));
        if let Some(ip) = ip {
            request = request.header("x-real-ip", ip);
        }
        let (response, _) = client
            .send_request(request.body(()).unwrap(), true)
            .unwrap();
        assert_eq!(
            timeout(Duration::from_secs(1), response)
                .await
                .unwrap()
                .unwrap()
                .status(),
            expected
        );
    }
    server.server.shutdown();
    timeout(Duration::from_secs(1), connection_task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
