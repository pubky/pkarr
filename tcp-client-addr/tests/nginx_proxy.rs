//! Optional end-to-end test with a real Nginx stream proxy.
//! Run with `cargo test -p tcp-client-addr --test nginx_proxy -- --ignored`.

use std::{
    fs,
    net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener},
    path::PathBuf,
    process::Command,
    time::Duration,
};

use tcp_client_addr::{IdentityMode, ProxyProtocol};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::{sleep, timeout},
};

struct NginxProxy {
    container_id: String,
    config_path: PathBuf,
}

impl NginxProxy {
    fn start(frontend: SocketAddr, backend: SocketAddr) -> Self {
        let config_path = std::env::temp_dir().join(format!(
            "client-identity-nginx-{}-{}.conf",
            std::process::id(),
            frontend.port()
        ));
        let config = format!(
            "events {{}}\nstream {{ server {{ listen {frontend}; proxy_pass {backend}; proxy_protocol on; }} }}\n"
        );
        fs::write(&config_path, config).unwrap();

        let output = Command::new("docker")
            .args(["run", "--rm", "-d", "--network", "host", "--mount"])
            .arg(format!(
                "type=bind,source={},target=/etc/nginx/nginx.conf,readonly",
                config_path.display()
            ))
            .arg("nginx:alpine")
            .output()
            .unwrap_or_else(|error| {
                let _ = fs::remove_file(&config_path);
                panic!("Docker CLI is required for this ignored test: {error}");
            });
        if !output.status.success() {
            let _ = fs::remove_file(&config_path);
            panic!(
                "failed to start Nginx: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Self {
            container_id: String::from_utf8(output.stdout).unwrap().trim().to_owned(),
            config_path,
        }
    }
}

impl Drop for NginxProxy {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["stop", &self.container_id])
            .output();
        let _ = fs::remove_file(&self.config_path);
    }
}

#[tokio::test]
#[ignore = "requires Docker, a cached nginx:alpine image, and host networking"]
async fn nginx_stream_sends_real_client_address_and_preserves_application_bytes() {
    let backend = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let frontend = StdTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let frontend_address = frontend.local_addr().unwrap();
    drop(frontend);

    let _proxy = NginxProxy::start(frontend_address, backend.local_addr().unwrap());
    let mut client = timeout(Duration::from_secs(5), async {
        loop {
            match TcpStream::connect(frontend_address).await {
                Ok(stream) => break stream,
                Err(_) => sleep(Duration::from_millis(50)).await,
            }
        }
    })
    .await
    .expect("Nginx did not start listening");
    let client_address = client.local_addr().unwrap();
    client.write_all(b"hello").await.unwrap();

    let (stream, _) = timeout(Duration::from_secs(5), backend.accept())
        .await
        .unwrap()
        .unwrap();
    let mode =
        IdentityMode::ProxyProtocol(ProxyProtocol::new(["127.0.0.1/32".parse().unwrap()]).unwrap());
    let (mut stream, identity) = mode.identify(stream).await.unwrap();
    assert_eq!(identity.client(), client_address);
    assert_ne!(identity.peer().port(), identity.client().port());
    let mut body = [0; 5];
    stream.read_exact(&mut body).await.unwrap();
    assert_eq!(&body, b"hello");
}
