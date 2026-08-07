use std::{
    io,
    net::{Ipv4Addr, SocketAddr, TcpListener},
    sync::Arc,
    time::Duration,
};

use axum::Router;
use h2::Ping;
use http::{StatusCode, Version};
use pkarr::Keypair;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::Notify,
    task::JoinHandle,
    time::{sleep, timeout},
};
use tower_http::timeout::RequestBodyDeadlineLayer;

use crate::http_server;

#[tokio::test]
async fn idle_connection_is_closed_after_request_deadline() {
    let server = run_http_test_server(Duration::from_millis(50), Router::new());
    let mut stream = TcpStream::connect(server.address).await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("idle connection should time out")
        .unwrap();
}

#[tokio::test]
async fn incomplete_first_request_is_closed_after_deadline() {
    let server = run_http_test_server(Duration::from_millis(50), Router::new());
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream.write_all(b"GET / HTTP/1.1\r\nHost:").await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("incomplete headers should time out")
        .unwrap();
}

#[tokio::test]
async fn incomplete_second_http1_request_is_closed_after_header_timeout() {
    let app = Router::new().route("/", axum::routing::get(|| async { StatusCode::NO_CONTENT }));
    let server = run_http_test_server(Duration::from_millis(50), app);
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    wait_for_http1_response(&mut stream).await.unwrap();

    stream.write_all(b"GET / HTTP/1.1\r\nHost:").await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("incomplete second request should time out")
        .unwrap();
}

#[tokio::test]
async fn partial_http2_preface_is_closed_after_request_deadline() {
    let server = run_http_test_server(Duration::from_millis(50), Router::new());
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream.write_all(b"PRI * HTTP/2.0\r\n").await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("partial HTTP/2 preface should time out")
        .unwrap();
}

#[tokio::test]
async fn http2_handshake_without_request_is_closed_after_request_deadline() {
    let server = run_http_test_server(Duration::from_millis(50), Router::new());
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    stream
        .write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
        .await
        .unwrap();
    stream
        .write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0])
        .await
        .unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("HTTP/2 connection without a request should time out")
        .unwrap();
}

#[tokio::test]
async fn http2_prior_knowledge_requests_succeed_before_idle_deadline() {
    let app = Router::new().route("/", axum::routing::get(|| async { StatusCode::NO_CONTENT }));
    let server = run_http_test_server(Duration::from_secs(1), app);
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();
    let response = response.await.unwrap();

    assert_eq!(response.version(), Version::HTTP_2);
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    sleep(Duration::from_millis(100)).await;

    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();
    let response = response.await.unwrap();

    assert_eq!(response.version(), Version::HTTP_2);
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    connection_task.abort();
}

#[tokio::test]
async fn http2_active_request_may_exceed_idle_deadline() {
    let app = Router::new().route(
        "/",
        axum::routing::get(|| async {
            sleep(Duration::from_millis(1_100)).await;
            StatusCode::NO_CONTENT
        }),
    );
    let server = run_http_test_server(Duration::from_secs(1), app);
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();

    let response = timeout(Duration::from_secs(2), response)
        .await
        .expect("active HTTP/2 request should outlive the idle deadline")
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    connection_task.abort();
}

#[tokio::test]
async fn http2_deadline_remains_disarmed_until_all_requests_finish() {
    let slow_request_started = Arc::new(Notify::new());
    let handler_notification = Arc::clone(&slow_request_started);
    let app = Router::new()
        .route(
            "/slow",
            axum::routing::get(move || {
                let notification = Arc::clone(&handler_notification);
                async move {
                    notification.notify_one();
                    sleep(Duration::from_millis(500)).await;
                    StatusCode::NO_CONTENT
                }
            }),
        )
        .route(
            "/fast",
            axum::routing::get(|| async { StatusCode::NO_CONTENT }),
        );
    let server = run_http_test_server(Duration::from_millis(200), app);
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);

    let slow_request = http::Request::get(format!("http://{}/slow", server.address))
        .body(())
        .unwrap();
    let (slow_response, _) = client.send_request(slow_request, true).unwrap();
    slow_request_started.notified().await;

    client = client.ready().await.unwrap();
    let fast_request = http::Request::get(format!("http://{}/fast", server.address))
        .body(())
        .unwrap();
    let (fast_response, _) = client.send_request(fast_request, true).unwrap();
    assert_eq!(
        fast_response.await.unwrap().status(),
        StatusCode::NO_CONTENT
    );

    let slow_response = timeout(Duration::from_secs(1), slow_response)
        .await
        .expect("the remaining active request should keep the deadline disarmed")
        .unwrap();
    assert_eq!(slow_response.status(), StatusCode::NO_CONTENT);

    connection_task.abort();
}

#[tokio::test]
async fn http2_connection_closes_after_request_deadline_despite_ping_traffic() {
    let app = Router::new().route("/", axum::routing::get(|| async { StatusCode::NO_CONTENT }));
    let server = run_http_test_server(Duration::from_millis(100), app);
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, mut connection) = h2::client::handshake(stream).await.unwrap();
    let mut ping_pong = connection
        .ping_pong()
        .expect("the client connection should expose a PING handle");
    let connection_task = tokio::spawn(connection);
    let request = http::Request::get(format!("http://{}/", server.address))
        .body(())
        .unwrap();
    let (response, _) = client.send_request(request, true).unwrap();
    let response = response.await.unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let ping_task = tokio::spawn(async move {
        loop {
            sleep(Duration::from_millis(20)).await;
            if ping_pong.ping(Ping::opaque()).await.is_err() {
                break;
            }
        }
    });

    let _connection_result = timeout(Duration::from_secs(1), connection_task)
        .await
        .expect("idle HTTP/2 connection should time out")
        .expect("HTTP/2 client connection task should not panic");

    ping_task.abort();
    drop(client);
}

#[tokio::test]
async fn incomplete_request_body_is_closed_after_body_deadline() {
    let app = Router::new()
        .route("/{key}", axum::routing::put(|_: bytes::Bytes| async {}))
        .layer(RequestBodyDeadlineLayer::new(Duration::from_millis(50)));
    let server = run_http_test_server(Duration::from_secs(1), app);
    let mut stream = TcpStream::connect(server.address).await.unwrap();
    let public_key = Keypair::random().public_key().to_string();
    let headers =
        format!("PUT /{public_key} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\n\r\n");
    stream.write_all(headers.as_bytes()).await.unwrap();

    timeout(Duration::from_secs(1), wait_for_disconnect(&mut stream))
        .await
        .expect("incomplete request body should time out")
        .unwrap();
}

#[tokio::test]
async fn incomplete_http2_request_body_hits_body_deadline() {
    let app = Router::new()
        .route("/{key}", axum::routing::put(|_: bytes::Bytes| async {}))
        .layer(RequestBodyDeadlineLayer::new(Duration::from_millis(50)));
    let server = run_http_test_server(Duration::from_secs(1), app);
    let stream = TcpStream::connect(server.address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(stream).await.unwrap();
    let connection_task = tokio::spawn(connection);
    let public_key = Keypair::random().public_key().to_string();
    let request = http::Request::put(format!("http://{}/{public_key}", server.address))
        .body(())
        .unwrap();
    let (response, stalled_body) = client.send_request(request, false).unwrap();

    let response = timeout(Duration::from_secs(1), response)
        .await
        .expect("incomplete HTTP/2 body should time out")
        .unwrap();

    assert!(response.status().is_client_error());

    drop(stalled_body);
    connection_task.abort();
}

struct TestServer {
    address: SocketAddr,
    task: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn run_http_test_server(request_read_timeout: Duration, app: Router) -> TestServer {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let server = http_server(listener, request_read_timeout).unwrap();
    let task = tokio::spawn(async move {
        server.serve(app.into_make_service()).await.unwrap();
    });

    TestServer { address, task }
}

async fn wait_for_http1_response(stream: &mut TcpStream) -> io::Result<()> {
    let mut response = Vec::new();
    let mut buffer = [0; 1024];

    while !response.windows(4).any(|window| window == b"\r\n\r\n") {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed before the response arrived",
            ));
        }
        response.extend_from_slice(&buffer[..read]);
    }

    Ok(())
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
