//! A server that functions as a [pkarr](https://github.com/pubky/pkarr/) [relay](https://github.com/pubky/pkarr/blob/main/design/relays.md).
//!
//! You can run this relay as a binary or a crate for testing purposes.
//!

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod config;
mod dht_service;
mod error;
mod extractors;
mod handlers;
mod index;
mod quota_config;
mod rate_limiting;
mod real_ip;
mod response;

use std::{
    net::{SocketAddr, SocketAddrV4, TcpListener, ToSocketAddrs},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::anyhow;
use axum::{extract::DefaultBodyLimit, http::HeaderName, Router};
use axum_server::Handle;

use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

use pkarr::{
    dht::{DhtClient, ReportPolicy},
    extra::lmdb_cache::LmdbCache,
    mainline, Timestamp, PKARR_DHT_STORED_NODES, PKARR_INVALID_SIGNED_PACKET_SEQ,
};
use url::Url;

use config::{RelayConfig, CACHE_DIR};
use dht_service::DhtService;

pub use quota_config::{RequestCountQuota, TimeUnit};
pub use rate_limiting::RateLimiterConfig;

/// A builder for Pkarr [Relay]
pub struct RelayBuilder {
    config: RelayConfig,
    dht: mainline::Config,
    report_policy: ReportPolicy,
}

impl RelayBuilder {
    /// Set the port for the HTTP endpoint.
    pub fn http_port(&mut self, port: u16) -> &mut Self {
        self.config.http.port = port;

        self
    }

    /// Set the storage directory.
    ///
    /// This Relay's cache will be stored in a subdirectory (`pkarr-cache`) inside
    /// that storage directory
    ///
    /// Defaults to the path to the user's data directory
    pub fn storage(&mut self, storage: PathBuf) -> &mut Self {
        self.config.cache.path = Some(storage.join(CACHE_DIR));

        self
    }

    /// See [pkarr::ClientBuilder::cache_size]
    ///
    /// Defaults to `1_000_000`
    pub fn cache_size(&mut self, size: usize) -> &mut Self {
        self.config.cache.size = size;

        self
    }

    /// Disable the rate limiter.
    ///
    /// Useful when running in a local test network.
    pub fn disable_rate_limiter(&mut self) -> &mut Self {
        self.config.rate_limiter = None;

        self
    }

    /// Set the [RateLimiterConfig].
    ///
    /// Defaults to [RateLimiterConfig::default].
    pub fn rate_limiter_config(&mut self, config: RateLimiterConfig) -> &mut Self {
        self.config.rate_limiter = Some(config);

        self
    }

    /// Set the policy used to classify DHT query diagnostics.
    ///
    /// Defaults to [`ReportPolicy::mainnet`].
    pub fn report_policy(&mut self, policy: ReportPolicy) -> &mut Self {
        self.report_policy = policy;

        self
    }

    /// Allows mutating the internal [pkarr::mainline] DHT configuration.
    pub fn dht<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&mut pkarr::mainline::Config) -> &mut pkarr::mainline::Config,
    {
        f(&mut self.dht);

        self
    }

    /// Set the maximum request timeout for DHT requests.
    pub fn request_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.dht.request_timeout = timeout;

        self
    }

    /// Run a Pkarr relay with the provided configuration.
    ///
    /// # Safety
    /// This method is marked as unsafe because it uses `LmdbCache`, which can lead to
    /// undefined behavior if the lock file is corrupted or improperly handled.
    pub async unsafe fn run(self) -> anyhow::Result<Relay> {
        unsafe { Relay::run_with_dht(self.config, self.dht, self.report_policy) }.await
    }
}

/// A running instance of a Pkarr relay server.
///
/// This struct represents a running relay server and provides methods to interact with it,
/// such as retrieving the server's address or shutting it down.
pub struct Relay {
    handle: Handle<SocketAddr>,
    relay_address: SocketAddr,
}

impl Relay {
    /// Run a Pkarr relay with the provided configuration.
    ///
    /// # Safety
    /// This method is marked as unsafe because it uses `LmdbCache`, which can lead to
    /// undefined behavior if the lock file is corrupted or improperly handled.
    ///
    /// # Arguments
    /// * `config` - The configuration for the relay.
    ///
    /// # Returns
    /// A `Result` containing the `Relay` instance or an error.
    async unsafe fn run(config: RelayConfig) -> anyhow::Result<Self> {
        let dht_config = mainline_config(&config);

        unsafe { Self::run_with_dht(config, dht_config, ReportPolicy::mainnet()) }.await
    }

    async unsafe fn run_with_dht(
        config: RelayConfig,
        mut dht_config: mainline::Config,
        report_policy: ReportPolicy,
    ) -> anyhow::Result<Self> {
        tracing::debug!(?config, "Pkarr server config");

        let cache_path = config
            .cache
            .path
            .unwrap_or_else(|| {
                tracing::warn!("Cache path is not configured, running ephemeral Relay");
                std::env::temp_dir().join(Timestamp::now().to_string())
            })
            .join(CACHE_DIR);

        let cache = Arc::new(LmdbCache::open(&cache_path, config.cache.size)?);

        let rate_limiters =
            build_rate_limiters(config.rate_limiter.as_ref(), &mut dht_config).await?;

        let dht_client = DhtClient::build(dht_config)?;

        let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], config.http.port)))?;
        // On axum-server 0.8.0 the `.set_nonblocking(true)` call does not take place internally anymore
        // See open issue https://github.com/programatik29/axum-server/issues/181
        listener.set_nonblocking(true)?;

        let node_address = dht_client.info().await.local_addr();
        let relay_address = listener.local_addr()?;

        info!("Cache path: {cache_path:?}");
        info!("Running as a DHT node on {node_address}");
        info!("Running as a relay on TCP socket {relay_address}");

        let dht = DhtService::new(
            config.cache.minimum_ttl,
            config.cache.maximum_ttl,
            cache,
            dht_client,
            report_policy,
            rate_limiters.user_dht,
        );
        let state = AppState { dht };
        let app = create_app(state, rate_limiters.http, rate_limiters.behind_proxy);

        let handle = Handle::new();

        let task = axum_server::from_tcp(listener)?
            .handle(handle.clone())
            .serve(app.into_make_service_with_connect_info::<SocketAddr>());

        tokio::spawn(task);

        Ok(Relay {
            handle,
            relay_address,
        })
    }

    /// Create a builder for running a [Relay]
    pub fn builder() -> RelayBuilder {
        RelayBuilder {
            config: Default::default(),
            dht: Default::default(),
            report_policy: ReportPolicy::mainnet(),
        }
    }

    /// Run a [Relay] with a configuration file path.
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn run_with_config_file(
        config_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        unsafe { Self::run(RelayConfig::load(config_path).await?) }.await
    }

    /// Run an ephemeral Pkarr relay on a random port number for testing purposes.
    /// Binds to `127.0.0.1`.
    /// # Arguments
    /// * `testnet` - A reference to a `mainline::Testnet` for bootstrapping the DHT.
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async fn run_test(testnet: &pkarr::mainline::Testnet) -> anyhow::Result<Self> {
        let config = RelayConfig {
            http: config::HttpConfig { port: 0 },
            rate_limiter: None,
            ..Default::default()
        };

        let mut dht_config = mainline_config(&config);
        dht_config.bootstrap = Some(to_socket_address_v4(&testnet.bootstrap));
        dht_config.request_timeout = Duration::from_millis(100);
        dht_config.server_mode = true;
        dht_config.bind_address = Some(std::net::Ipv4Addr::LOCALHOST);

        Ok(unsafe { Self::run_with_dht(config, dht_config, ReportPolicy::testnet()).await? })
    }

    /// Run a Pkarr relay in a Testnet mode (on port 15411).
    ///
    /// # Safety
    /// Homeserver uses LMDB, opening which is marked [unsafe](https://docs.rs/heed/latest/heed/struct.EnvOpenOptions.html#safety-1),
    /// because the possible Undefined Behavior (UB) if the lock file is broken.
    pub async unsafe fn run_testnet() -> anyhow::Result<Self> {
        let testnet = pkarr::mainline::Testnet::builder(10).build()?;

        // Leaking the testnet to avoid dropping and shutting them down.
        for node in testnet.nodes {
            Box::leak(Box::new(node));
        }

        let config = RelayConfig {
            http: config::HttpConfig { port: 15411 },
            rate_limiter: None,
            ..Default::default()
        };

        let mut dht_config = mainline_config(&config);
        dht_config.bootstrap = Some(to_socket_address_v4(&testnet.bootstrap));
        dht_config.request_timeout = Duration::from_millis(100);
        dht_config.server_mode = true;
        dht_config.bind_address = Some(std::net::Ipv4Addr::LOCALHOST);

        Self::run_with_dht(config, dht_config, ReportPolicy::testnet()).await
    }

    /// Returns the HTTP socket address of the relay.
    pub fn relay_address(&self) -> SocketAddr {
        self.relay_address
    }

    /// Returns the localhost URL of the HTTP server.
    pub fn local_url(&self) -> Url {
        Url::parse(&format!("http://localhost:{}", self.relay_address.port()))
            .expect("local_url should be formatted fine")
    }

    /// Shutdown the relay server.
    pub fn shutdown(&self) {
        self.handle.shutdown();
    }
}

fn mainline_config(config: &RelayConfig) -> mainline::Config {
    mainline::Config {
        port: config.mainline.port,
        ..Default::default()
    }
}

struct RateLimiters {
    http: Option<rate_limiting::IpRateLimiter>,
    user_dht: Option<rate_limiting::UserDhtRateLimiter>,
    behind_proxy: bool,
}

async fn build_rate_limiters(
    rate_limiter_config: Option<&RateLimiterConfig>,
    dht_config: &mut mainline::Config,
) -> anyhow::Result<RateLimiters> {
    let Some(rate_limiter_config) = rate_limiter_config else {
        return Ok(RateLimiters {
            http: None,
            user_dht: None,
            behind_proxy: false,
        });
    };

    let http = rate_limiting::IpRateLimiter::new(rate_limiter_config)
        .await
        .map_err(|e| anyhow!("Failed to build IpRateLimiter: {e}"))?;
    let user_dht = rate_limiting::UserDhtRateLimiter::new(rate_limiter_config)
        .await
        .map_err(|e| anyhow!("Failed to build UserDhtRateLimiter: {e}"))?;
    let dht = rate_limiting::DhtRateLimiter::new(rate_limiter_config)
        .await
        .map_err(|e| anyhow!("Failed to build DhtRateLimiter: {e}"))?;

    dht_config.server_settings = pkarr::mainline::ServerSettings {
        filter: Box::new(dht),
        ..Default::default()
    };

    Ok(RateLimiters {
        http: Some(http),
        user_dht: Some(user_dht),
        behind_proxy: rate_limiter_config.behind_proxy,
    })
}

fn create_app(
    state: AppState,
    rate_limiter: Option<rate_limiting::IpRateLimiter>,
    behind_proxy: bool,
) -> Router {
    let mut router = Router::new()
        .route(
            "/{key}",
            axum::routing::get(crate::handlers::get).put(crate::handlers::put),
        )
        .route("/", axum::routing::get(crate::handlers::index))
        .with_state(state)
        .layer(DefaultBodyLimit::max(1104))
        .layer(cors_layer())
        .layer(TraceLayer::new_for_http());

    if let Some(rate_limiter) = rate_limiter {
        router = rate_limiter
            .layer(router)
            .layer(real_ip::middleware(behind_proxy));
    }

    router
}

fn cors_layer() -> CorsLayer {
    let invalid_signed_packet_seq_header =
        HeaderName::from_bytes(PKARR_INVALID_SIGNED_PACKET_SEQ.as_bytes())
            .expect("Pkarr invalid signed packet seq header name is valid");
    let dht_stored_nodes_header = HeaderName::from_bytes(PKARR_DHT_STORED_NODES.as_bytes())
        .expect("Pkarr DHT stored nodes header name is valid");

    CorsLayer::very_permissive()
        .expose_headers([invalid_signed_packet_seq_header, dht_stored_nodes_header])
}

fn to_socket_address_v4<T: ToSocketAddrs>(bootstrap: &[T]) -> Vec<SocketAddrV4> {
    bootstrap
        .iter()
        .flat_map(|address| {
            address.to_socket_addrs().map(|addresses| {
                addresses
                    .filter_map(|address| match address {
                        SocketAddr::V4(address) => Some(address),
                        SocketAddr::V6(_) => None,
                    })
                    .collect::<Box<[_]>>()
            })
        })
        .flatten()
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) struct AppState {
    pub(crate) dht: DhtService,
}

#[cfg(test)]
mod tests {
    use std::{net::Ipv4Addr, num::NonZeroU32, time::Duration};

    use http::StatusCode;
    use pkarr::{
        Keypair, SignedPacket, Timestamp, PKARR_DHT_STORED_NODES, PKARR_INVALID_SIGNED_PACKET_SEQ,
    };

    use super::{to_socket_address_v4, RateLimiterConfig, Relay, RequestCountQuota};

    #[tokio::test]
    async fn cors_exposes_invalid_signed_packet_seq_header() {
        let testnet = pkarr::mainline::Testnet::builder(2).build().unwrap();
        let relay = run_rate_limited_relay(&testnet).await;

        let response = reqwest::Client::new()
            .get(relay.local_url())
            .header("Origin", "https://example.com")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let exposed_headers = response
            .headers()
            .get("access-control-expose-headers")
            .unwrap()
            .to_str()
            .unwrap();

        assert!(exposed_headers.split(',').any(|header| header
            .trim()
            .eq_ignore_ascii_case(PKARR_INVALID_SIGNED_PACKET_SEQ)));
        assert!(exposed_headers
            .split(',')
            .any(|header| header.trim().eq_ignore_ascii_case(PKARR_DHT_STORED_NODES)));

        relay.shutdown();
    }

    #[tokio::test]
    async fn user_dht_rate_limit_only_blocks_dht_operations() {
        let testnet = pkarr::mainline::Testnet::builder(2).build().unwrap();
        let relay = run_rate_limited_relay(&testnet).await;
        let http = reqwest::Client::new();
        let keypair = Keypair::random();
        let signed_packet = signed_packet(&keypair);
        let relay_url = relay.local_url();

        let put_response = http
            .put(relay_url.join(&keypair.public_key().to_string()).unwrap())
            .body(signed_packet.to_relay_payload())
            .send()
            .await
            .unwrap();
        assert_eq!(put_response.status(), StatusCode::NO_CONTENT);
        put_response
            .headers()
            .get(PKARR_DHT_STORED_NODES)
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<u32>()
            .unwrap();

        let local_cache_response = http
            .get(
                relay_url
                    .join(&format!("{}?policy=CacheOnly", keypair.public_key()))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(local_cache_response.status(), StatusCode::OK);

        let cache_first_response = http
            .get(
                relay_url
                    .join(&format!("{}?policy=CacheFirst", keypair.public_key()))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(cache_first_response.status(), StatusCode::OK);

        let dht_response = http
            .get(
                relay_url
                    .join(&format!("{}?policy=NetworkOnly", keypair.public_key()))
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(dht_response.status(), StatusCode::TOO_MANY_REQUESTS);

        relay.shutdown();
    }

    async fn run_rate_limited_relay(testnet: &pkarr::mainline::Testnet) -> Relay {
        let storage = std::env::temp_dir().join(format!("pkarr-relay-{}", Timestamp::now()));
        let mut builder = Relay::builder();
        builder
            .cache_size(10)
            .http_port(0)
            .storage(storage)
            .rate_limiter_config(RateLimiterConfig {
                behind_proxy: false,
                quota: "60r/s".parse::<RequestCountQuota>().unwrap(),
                burst: NonZeroU32::new(100),
                dht_quota: "60r/s".parse::<RequestCountQuota>().unwrap(),
                dht_burst: NonZeroU32::new(100),
                user_dht_quota: "1r/h".parse::<RequestCountQuota>().unwrap(),
                user_dht_burst: NonZeroU32::new(1),
            })
            .request_timeout(Duration::from_millis(100))
            .dht(|config| {
                config.bootstrap = Some(to_socket_address_v4(&testnet.bootstrap));
                config.bind_address = Some(Ipv4Addr::LOCALHOST);
                config
            });

        unsafe { builder.run().await.unwrap() }
    }

    fn signed_packet(keypair: &Keypair) -> SignedPacket {
        SignedPacket::builder()
            .txt(
                "example.com".try_into().unwrap(),
                "rate-limited".try_into().unwrap(),
                300,
            )
            .sign(keypair)
            .unwrap()
    }
}
