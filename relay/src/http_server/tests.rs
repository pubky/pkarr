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
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = HttpServer::spawn_with_limits(listener, app, limits).unwrap();

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
