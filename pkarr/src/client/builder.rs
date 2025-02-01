use std::{
    net::{SocketAddr, SocketAddrV4, ToSocketAddrs},
    sync::Arc,
    time::Duration,
};

#[cfg(feature = "relays")]
use url::Url;

#[cfg(feature = "dht")]
use mainline::rpc::DEFAULT_REQUEST_TIMEOUT;

use crate::{
    Cache, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL, DEFAULT_RESOLVERS,
};

use super::native::{BuildError, Client};

/// [Client]'s Config
#[derive(Clone)]
pub struct Config {
    /// Defaults to [DEFAULT_CACHE_SIZE]
    pub cache_size: usize,
    /// Used in the `min` parameter in [crate::SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MINIMUM_TTL]
    pub minimum_ttl: u32,
    /// Used in the `max` parameter in [crate::SignedPacket::expires_in].
    ///
    /// Defaults to [DEFAULT_MAXIMUM_TTL]
    pub maximum_ttl: u32,
    /// Custom [Cache] implementation, defaults to [crate::InMemoryCache]
    pub cache: Option<Arc<dyn Cache>>,

    pub dht: Option<mainline::DhtBuilder>,
    /// A set of [resolver](https://pkarr.org/resolvers)s
    /// to be queried alongside the Dht routing table, to
    /// lower the latency on cold starts, and help if the
    /// Dht is missing values not't republished often enough.
    ///
    /// Defaults to [DEFAULT_RESOLVERS]
    pub resolvers: Option<Vec<SocketAddrV4>>,

    /// Pkarr [Relays](https://pkarr.org/relays) Urls
    #[cfg(feature = "relays")]
    pub relays: Option<Vec<Url>>,
    /// Tokio runtime to use in relyas client.

    /// Timeout for both Dht and Relays requests.
    ///
    /// The longer this timeout the longer resolve queries will take before consider failed.
    ///
    /// Defaults to [mainline::rpc::DEFAULT_REQUEST_TIMEOUT]
    pub request_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cache_size: DEFAULT_CACHE_SIZE,
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            cache: None,

            #[cfg(feature = "dht")]
            dht: Some(mainline::Dht::builder()),
            #[cfg(feature = "dht")]
            resolvers: Some(resolvers_to_socket_addrs(&DEFAULT_RESOLVERS)),

            #[cfg(feature = "relays")]
            relays: Some(
                crate::DEFAULT_RELAYS
                    .iter()
                    .map(|s| {
                        Url::parse(s).expect("DEFAULT_RELAYS should be parsed to Url successfully.")
                    })
                    .collect(),
            ),

            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Config");

        debug_struct.field("cache_size", &self.cache_size);
        debug_struct.field("minimum_ttl", &self.minimum_ttl);
        debug_struct.field("maximum_ttl", &self.maximum_ttl);
        debug_struct.field("cache", &self.cache);

        #[cfg(feature = "dht")]
        debug_struct.field("dht", &self.dht);
        #[cfg(feature = "dht")]
        debug_struct.field("resolvers", &self.resolvers);

        #[cfg(feature = "relays")]
        debug_struct.field(
            "relays",
            &self
                .relays
                .as_ref()
                .map(|urls| urls.iter().map(|url| url.as_str()).collect::<Vec<_>>()),
        );

        debug_struct.field("request_timeout", &self.request_timeout);

        debug_struct.finish()
    }
}

#[derive(Debug, Default)]
pub struct ClientBuilder(Config);

impl ClientBuilder {
    /// Similar to crates `no-default-features`, this method will remove the default
    /// [mainline::Config::bootstrap], [Config::resolvers], and [Config::relays],
    /// effectively disabling the use of both [mainline] and [Relays](https://pkarr.org/relays).
    ///
    /// You can [Self::use_mainline] or [Self::use_relays] to add both or either
    /// back with the default configurations for each.
    ///
    /// Or you can use [Self::relays] to use custom [Relays](https://pkarr.org/relays).
    ///
    /// Similarly you can use [Self::resolvers] and / or [Self::bootstrap] to use [mainline]
    /// with custom configurations.
    pub fn no_default_network(&mut self) -> &mut Self {
        #[cfg(feature = "dht")]
        {
            self.0.resolvers = None;
            self.0.dht = None;
        }
        #[cfg(feature = "relays")]
        {
            self.0.relays = None;
        }

        self
    }

    /// Reenable using [mainline] DHT with [DEFAULT_RESOLVERS] and [mainline::Config::default].
    #[cfg(feature = "dht")]
    pub fn use_dht(&mut self) -> &mut Self {
        self.0.resolvers = Some(resolvers_to_socket_addrs(&DEFAULT_RESOLVERS));
        self.0.dht = Some(mainline::DhtBuilder::default());

        self
    }

    /// Disable relays, and use the Dht only.
    ///
    /// Equivilant to `builder.no_default_network().use_relays();`
    pub fn no_dht(&mut self) -> &mut Self {
        self.no_default_network();
        #[cfg(feature = "relays")]
        self.use_relays();

        self
    }

    /// Convienent method to set the [mainline::Config::bootstrap] in [Config::dht].
    ///
    /// You can start a separate Dht network by setting this to an empty array.
    ///
    /// If you want to extend [Config::dht_config::bootstrap][mainline::Config::bootstrap] nodes with more nodes, you can
    /// use [Self::extra_bootstrap].
    #[cfg(feature = "dht")]
    pub fn bootstrap(&mut self, bootstrap: &[String]) -> &mut Self {
        self.dht(|b| b.bootstrap(bootstrap));

        self
    }

    /// Create [Self::dht] if `None`, and allows mutating it with a callback function.
    #[cfg(feature = "dht")]
    pub fn dht<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&mut mainline::DhtBuilder) -> &mut mainline::DhtBuilder,
    {
        if self.0.dht.is_none() {
            self.0.dht = Some(Default::default());
        }

        if let Some(ref mut builder) = self.0.dht {
            f(builder);
        };

        self
    }

    #[cfg(feature = "dht")]
    /// Extend the [Config::dht_config::bootstrap][mainline::Config::bootstrap] nodes.
    ///
    /// If you want to set (override) the [Config::dht_config::bootsrtap][mainline::Config::bootstrap],
    /// use [Self::bootstrap]
    pub fn extra_resolvers(&mut self, resolvers: Vec<String>) -> &mut Self {
        let resolvers = resolvers_to_socket_addrs(&resolvers);

        if let Some(ref mut existing) = self.0.resolvers {
            existing.extend_from_slice(&resolvers);
        };

        self
    }

    /// Set custom set of [resolvers](Config::resolvers).
    ///
    /// You can disable using resolvers by passing `None`.
    ///
    /// If you want to extend the [Config::resolvers] with more nodes, you can
    /// use [Self::extra_resolvers].
    #[cfg(feature = "dht")]
    pub fn resolvers(&mut self, resolvers: Option<Vec<String>>) -> &mut Self {
        self.0.resolvers = resolvers.map(|resolvers| resolvers_to_socket_addrs(&resolvers));

        self
    }

    #[cfg(feature = "dht")]
    /// Extend the current [Config::resolvers] with extra resolvers.
    ///
    /// If you want to set (override) the [Config::resolvers], use [Self::resolvers]
    pub fn extra_bootstrap(&mut self, bootstrap: &[String]) -> &mut Self {
        self.dht(|b| b.extra_bootstrap(bootstrap));

        self
    }

    #[cfg(feature = "relays")]
    /// Reenable using [Config::relays] with [crate::DEFAULT_RELAYS].
    pub fn use_relays(&mut self) -> &mut Self {
        self.0.relays = Some(
            crate::DEFAULT_RELAYS
                .iter()
                .map(|s| {
                    Url::parse(s).expect("DEFAULT_RELAYS should be parsed to Url successfully.")
                })
                .collect(),
        );

        self
    }

    /// Set custom set of [relays](Config::relays)
    #[cfg(feature = "relays")]
    pub fn relays(&mut self, relays: Option<Vec<Url>>) -> &mut Self {
        self.0.relays = relays;

        self
    }

    /// Disable relays, and use the Dht only.
    ///
    /// Equivilant to `builder.no_default_network().use_dht();`
    #[cfg(feature = "relays")]
    pub fn no_relays(&mut self) -> &mut Self {
        self.no_default_network();
        #[cfg(feature = "dht")]
        self.use_dht();

        self
    }

    #[cfg(feature = "relays")]
    /// Extend the current [Config::relays] with extra relays.
    ///
    /// If you want to set (override) the [Config::relays], use [Self::relays]
    pub fn extra_relays(&mut self, relays: Vec<Url>) -> &mut Self {
        if let Some(ref mut existing) = self.0.relays {
            for relay in relays {
                if !existing.contains(&relay) {
                    existing.push(relay)
                }
            }
        }

        self
    }

    /// Set the [Config::cache_size].
    ///
    /// Controls the capacity of [Cache].
    ///
    /// If set to `0` cache will be disabled.
    pub fn cache_size(&mut self, cache_size: usize) -> &mut Self {
        self.0.cache_size = cache_size;

        self
    }

    /// Set the [Config::minimum_ttl] value.
    ///
    /// Limits how soon a [crate::SignedPacket] is considered expired.
    pub fn minimum_ttl(&mut self, ttl: u32) -> &mut Self {
        self.0.minimum_ttl = ttl;

        self
    }

    /// Set the [Config::maximum_ttl] value.
    ///
    /// Limits how long it takes before a [crate::SignedPacket] is considered expired.
    pub fn maximum_ttl(&mut self, ttl: u32) -> &mut Self {
        self.0.maximum_ttl = ttl;

        self
    }

    /// Set a custom implementation of [Cache].
    pub fn cache(&mut self, cache: Arc<dyn Cache>) -> &mut Self {
        self.0.cache = Some(cache);

        self
    }

    /// Set the maximum [Config::request_timeout] for both Dht and relays client.
    ///
    /// Useful for testing NOT FOUND responses, where you want to reach the timeout
    /// sooner than the default of [mainline::rpc::DEFAULT_REQUEST_TIMEOUT].
    pub fn request_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.0.request_timeout = timeout;
        self.0.dht.as_mut().map(|b| b.request_timeout(timeout));

        self
    }

    pub fn build(&self) -> Result<Client, BuildError> {
        Client::new(self.0.clone())
    }
}

fn resolvers_to_socket_addrs<T: ToSocketAddrs>(resolvers: &[T]) -> Vec<SocketAddrV4> {
    resolvers
        .iter()
        .flat_map(|resolver| {
            resolver.to_socket_addrs().map(|iter| {
                iter.filter_map(|a| match a {
                    SocketAddr::V4(a) => Some(a),
                    _ => None,
                })
            })
        })
        .flatten()
        .collect::<Vec<_>>()
}
