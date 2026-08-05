use std::{
    fmt::Debug,
    net::{Ipv4Addr, SocketAddrV4},
    sync::Arc,
    time::Duration,
};

/// Filters incoming DHT requests.
pub trait RequestFilter: Debug + Send + Sync {
    /// Return whether a request from `from` should be handled.
    fn allow_request(&self, from: SocketAddrV4) -> bool;
}

/// Configuration for a [`super::DhtClient`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DhtConfig {
    /// UDP port used by the DHT node.
    ///
    /// When `None`, the DHT implementation's default port is used.
    pub port: Option<u16>,
    /// Whether the DHT node should answer incoming requests.
    pub server_mode: bool,
    /// Known public IPv4 address used to generate a BEP 42 secure node ID.
    ///
    /// When `None`, the DHT discovers its public address from peers.
    pub public_ip: Option<Ipv4Addr>,
    /// DHT nodes used to bootstrap discovery.
    ///
    /// When `None`, the DHT implementation's default bootstrap nodes are used.
    pub bootstrap: Option<Vec<SocketAddrV4>>,
    /// Local IPv4 address to bind the DHT socket to.
    ///
    /// When `None`, the operating system chooses the local address.
    pub bind_address: Option<Ipv4Addr>,
    /// Maximum duration for a DHT request.
    pub request_timeout: Duration,
    /// Filter for incoming DHT requests.
    pub request_filter: Option<Arc<dyn RequestFilter>>,
}

impl Default for DhtConfig {
    fn default() -> Self {
        Self {
            port: None,
            server_mode: false,
            public_ip: None,
            bootstrap: None,
            bind_address: None,
            request_timeout: crate::DEFAULT_REQUEST_TIMEOUT,
            request_filter: None,
        }
    }
}

impl DhtConfig {
    pub(super) fn into_mainline(self) -> mainline::Config {
        let mut config = mainline::Config {
            bootstrap: self.bootstrap,
            port: self.port,
            bind_address: self.bind_address,
            request_timeout: self.request_timeout,
            server_mode: self.server_mode,
            public_ip: self.public_ip,
            ..Default::default()
        };

        if let Some(filter) = self.request_filter {
            config.server_settings = mainline::ServerSettings {
                filter: Box::new(MainlineRequestFilter(filter)),
                ..Default::default()
            };
        }

        config
    }
}

#[derive(Clone, Debug)]
struct MainlineRequestFilter(Arc<dyn RequestFilter>);

impl MainlineRequestFilter {
    fn is_allowed(&self, from: SocketAddrV4) -> bool {
        self.0.allow_request(from)
    }
}

impl mainline::RequestFilter for MainlineRequestFilter {
    fn allow_request(&self, _request: &mainline::RequestSpecific, from: SocketAddrV4) -> bool {
        self.is_allowed(from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_client_configuration_to_mainline() {
        let bootstrap = "127.0.0.1:6881".parse().unwrap();
        let config = DhtConfig {
            port: Some(6881),
            server_mode: true,
            public_ip: Some("203.0.113.10".parse().unwrap()),
            bootstrap: Some(vec![bootstrap]),
            bind_address: Some(Ipv4Addr::LOCALHOST),
            request_timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let mainline = config.into_mainline();

        assert_eq!(mainline.port, Some(6881));
        assert_eq!(mainline.public_ip, Some("203.0.113.10".parse().unwrap()));
        assert_eq!(mainline.bootstrap, Some(vec![bootstrap]));
        assert_eq!(mainline.bind_address, Some(Ipv4Addr::LOCALHOST));
        assert_eq!(mainline.request_timeout, Duration::from_secs(5));
        assert!(mainline.server_mode);
    }

    #[derive(Debug)]
    struct LocalhostOnly;

    impl RequestFilter for LocalhostOnly {
        fn allow_request(&self, from: SocketAddrV4) -> bool {
            from.ip().is_loopback()
        }
    }

    #[test]
    fn mainline_request_filter_delegates_to_dht_filter() {
        let filter = MainlineRequestFilter(Arc::new(LocalhostOnly));

        assert!(filter.is_allowed("127.0.0.1:6881".parse().unwrap()));
        assert!(!filter.is_allowed("203.0.113.10:6881".parse().unwrap()));
    }
}
