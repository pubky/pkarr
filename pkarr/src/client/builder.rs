#[cfg(dht)]
use std::net::{SocketAddr, SocketAddrV4, ToSocketAddrs};
use std::{sync::Arc, time::Duration};

#[cfg(feature = "relays")]
use url::Url;

use crate::{Cache, DEFAULT_CACHE_SIZE, DEFAULT_MAXIMUM_TTL, DEFAULT_MINIMUM_TTL};

use crate::{errors::BuildError, Client};

#[cfg(feature = "endpoints")]
pub const DEFAULT_MAX_RECURSION_DEPTH: u8 = 7;

/// Default request timeout for DHT and relay requests.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// Configuration used to build a [`Client`].
#[derive(Clone)]
pub(crate) struct Config {
    /// Configures the [`crate::InMemoryCache`] size if no [`Self::cache`] is set.
    ///
    /// Defaults to [`DEFAULT_CACHE_SIZE`].
    pub cache_size: usize,
    /// Used as the `min` parameter in [`crate::SignedPacket::expires_in`].
    ///
    /// Defaults to [`DEFAULT_MINIMUM_TTL`].
    pub minimum_ttl: u32,
    /// Used as the `max` parameter in [`crate::SignedPacket::expires_in`].
    ///
    /// Defaults to [`DEFAULT_MAXIMUM_TTL`].
    pub maximum_ttl: u32,
    /// Custom [`Cache`] implementation. Defaults to [`crate::InMemoryCache`].
    pub cache: Option<Arc<dyn Cache>>,

    #[cfg(dht)]
    pub dht: Option<mainline::Config>,
    /// Policy used to classify DHT publish and resolve diagnostics.
    #[cfg(dht)]
    pub dht_report_policy: crate::dht::ReportPolicy,

    /// Pkarr [relay](https://github.com/pubky/pkarr/blob/main/design/relays.md) URLs.
    #[cfg(relays)]
    pub relays: Option<Vec<Url>>,
    /// Custom HTTP client used for relay requests.
    #[cfg(relays)]
    pub reqwest_client: Option<reqwest::Client>,

    /// Timeout for DHT and relay requests.
    ///
    /// A longer timeout allows requests more time to complete before they are considered failed.
    ///
    /// Defaults to [`DEFAULT_REQUEST_TIMEOUT`].
    pub request_timeout: Duration,

    #[cfg(feature = "endpoints")]
    pub max_recursion_depth: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cache_size: DEFAULT_CACHE_SIZE,
            minimum_ttl: DEFAULT_MINIMUM_TTL,
            maximum_ttl: DEFAULT_MAXIMUM_TTL,
            cache: None,

            #[cfg(dht)]
            dht: Some(make_dht_config(DEFAULT_REQUEST_TIMEOUT)),
            #[cfg(dht)]
            dht_report_policy: crate::dht::ReportPolicy::mainnet(),

            #[cfg(relays)]
            relays: Some(
                crate::DEFAULT_RELAYS
                    .iter()
                    .map(|s| {
                        Url::parse(s).expect("DEFAULT_RELAYS should be parsed to Url successfully.")
                    })
                    .collect(),
            ),
            #[cfg(relays)]
            reqwest_client: None,

            request_timeout: DEFAULT_REQUEST_TIMEOUT,

            #[cfg(feature = "endpoints")]
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
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

        #[cfg(dht)]
        debug_struct.field("dht", &self.dht);
        #[cfg(dht)]
        debug_struct.field("dht_report_policy", &self.dht_report_policy);
        #[cfg(dht)]
        #[cfg(relays)]
        debug_struct.field(
            "relays",
            &self
                .relays
                .as_ref()
                .map(|urls| urls.iter().map(|url| url.as_str()).collect::<Vec<_>>()),
        );
        #[cfg(relays)]
        debug_struct.field("reqwest_client", &self.reqwest_client);

        debug_struct.field("request_timeout", &self.request_timeout);

        debug_struct.finish()
    }
}

/// A builder for constructing a [`Client`] with custom configuration.
#[derive(Debug, Default, Clone)]
pub struct ClientBuilder(Config);

impl ClientBuilder {
    /// Similar to a crate's `--no-default-features` option, this method removes the
    /// default [`Self::bootstrap`] and [`Self::relays`], effectively disabling both
    /// [`mainline`] and [relays](https://github.com/pubky/pkarr/blob/main/design/relays.md).
    ///
    /// Use [`Self::relays`] to configure custom relays.
    ///
    /// Similarly, use [`Self::bootstrap`] or [`Self::dht`] to configure [`mainline`].
    pub fn no_default_network(&mut self) -> &mut Self {
        self.no_dht();
        self.no_relays();

        self
    }

    /// Disable the DHT and use relays only.
    pub fn no_dht(&mut self) -> &mut Self {
        #[cfg(dht)]
        {
            self.0.dht = None;
        }

        self
    }

    #[cfg(dht)]
    /// Creates a [`mainline::Config`] when absent and allows it to be mutated with a callback.
    pub fn dht<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&mut mainline::Config) -> &mut mainline::Config,
    {
        if self.0.dht.is_none() {
            self.0.dht = Some(make_dht_config(self.0.request_timeout));
        }

        if let Some(ref mut builder) = self.0.dht {
            f(builder);
        };

        self
    }

    /// Set the policy used to classify DHT publish and resolve diagnostics.
    #[cfg(dht)]
    pub fn dht_report_policy(&mut self, policy: crate::dht::ReportPolicy) -> &mut Self {
        self.0.dht_report_policy = policy;

        self
    }

    /// Convenience method for setting the `bootstrap` nodes in [`Self::dht`].
    ///
    /// You can start a separate DHT network by setting this to an empty array.
    ///
    /// If you want to extend bootstrap nodes with more nodes, you can
    /// use [`Self::extra_bootstrap`].
    #[cfg(dht)]
    pub fn bootstrap<T: ToSocketAddrs>(&mut self, bootstrap: &[T]) -> &mut Self {
        self.dht(|config| {
            config.bootstrap = Some(to_socket_address_v4(bootstrap));
            config
        });

        self
    }

    #[cfg(dht)]
    /// Extend the DHT bootstrapping nodes.
    ///
    /// If you want to set (override) the DHT bootstrapping nodes,
    /// use [`Self::bootstrap`] directly.
    pub fn extra_bootstrap<T: ToSocketAddrs>(&mut self, bootstrap: &[T]) -> &mut Self {
        self.dht(|config| {
            let mut existing = config.bootstrap.clone().unwrap_or_default();
            existing.extend(to_socket_address_v4(bootstrap));
            config.bootstrap = Some(existing);
            config
        });

        self
    }

    /// Disable relays and use only the DHT.
    pub fn no_relays(&mut self) -> &mut Self {
        #[cfg(feature = "relays")]
        {
            self.0.relays = None;
        }

        self
    }

    /// Set a custom set of [relays](https://github.com/pubky/pkarr/blob/main/design/relays.md).
    ///
    /// If you want to disable relays, use [`Self::no_relays`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidRelayUrl`] if a relay URL cannot be parsed or does not use
    /// the `http` or `https` scheme.
    #[cfg(feature = "relays")]
    pub fn relays<T: reqwest::IntoUrl + Clone>(
        &mut self,
        relays: &[T],
    ) -> Result<&mut Self, InvalidRelayUrl> {
        self.0.relays = Some(into_urls(relays)?);

        Ok(self)
    }

    #[cfg(feature = "relays")]
    /// Extend the current [`Self::relays`] with extra relays.
    ///
    /// If you want to replace the configured relays, use [`Self::relays`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidRelayUrl`] if a relay URL cannot be parsed or does not use
    /// the `http` or `https` scheme.
    pub fn extra_relays<T: reqwest::IntoUrl + Clone>(
        &mut self,
        relays: &[T],
    ) -> Result<&mut Self, InvalidRelayUrl> {
        if let Some(ref mut existing) = self.0.relays {
            for relay in into_urls(relays)? {
                if !existing.contains(&relay) {
                    existing.push(relay)
                }
            }
        }

        Ok(self)
    }

    /// Set the cache capacity.
    ///
    /// If set to `0`, the cache is disabled.
    pub fn cache_size(&mut self, cache_size: usize) -> &mut Self {
        self.0.cache_size = cache_size;

        self
    }

    /// Set the minimum TTL value.
    ///
    /// Limits how soon a [`crate::SignedPacket`] is considered expired.
    pub fn minimum_ttl(&mut self, ttl: u32) -> &mut Self {
        self.0.minimum_ttl = ttl;
        self.0.maximum_ttl = self.0.maximum_ttl.max(ttl);

        self
    }

    /// Set the maximum TTL value.
    ///
    /// Limits how long it takes before a [`crate::SignedPacket`] is considered expired.
    pub fn maximum_ttl(&mut self, ttl: u32) -> &mut Self {
        self.0.maximum_ttl = ttl;
        self.0.minimum_ttl = self.0.minimum_ttl.min(ttl);

        self
    }

    /// Set a custom implementation of [`Cache`].
    ///
    /// A cache with a capacity of `0` is treated as disabled.
    pub fn cache(&mut self, cache: Arc<dyn Cache>) -> &mut Self {
        self.0.cache = Some(cache);

        self
    }

    /// Set a custom [`reqwest::Client`] for relay HTTP requests.
    ///
    /// The client's request timeout still comes from [`Self::request_timeout`],
    /// because relay request timeouts are applied per request for native and
    /// WASM targets.
    #[cfg(relays)]
    pub fn reqwest_client(&mut self, client: reqwest::Client) -> &mut Self {
        self.0.reqwest_client = Some(client);

        self
    }

    /// Set the maximum timeout for DHT and relay requests.
    ///
    /// Useful for testing not-found responses when you want to reach the timeout
    /// sooner than the default of [`DEFAULT_REQUEST_TIMEOUT`].
    pub fn request_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.0.request_timeout = timeout;
        #[cfg(dht)]
        if let Some(config) = self.0.dht.as_mut() {
            config.request_timeout = timeout;
        }

        self
    }

    #[cfg(feature = "endpoints")]
    /// Set the maximum recursion depth for [endpoint](https://github.com/pubky/pkarr/blob/main/design/endpoints.md) resolution.
    ///
    /// Similar to BIND 9's [option](https://bind9.readthedocs.io/en/latest/reference.html#namedconf-statement-max-recursion-depth).
    ///
    /// Defaults to `7`.
    pub fn max_recursion_depth(&mut self, max_recursion_depth: u8) -> &mut Self {
        self.0.max_recursion_depth = max_recursion_depth;

        self
    }

    /// Build a [`Client`] with this builder's configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] if no network is configured or a configured backend
    /// cannot be constructed.
    pub fn build(&self) -> Result<Client, BuildError> {
        Client::new(self.0.clone())
    }
}

#[cfg(dht)]
fn make_dht_config(request_timeout: Duration) -> mainline::Config {
    mainline::Config {
        request_timeout,
        ..Default::default()
    }
}

#[cfg(dht)]
fn to_socket_address_v4<T: ToSocketAddrs>(bootstrap: &[T]) -> Vec<SocketAddrV4> {
    bootstrap
        .iter()
        .flat_map(|s| {
            s.to_socket_addrs().map(|addrs| {
                addrs
                    .filter_map(|addr| match addr {
                        SocketAddr::V4(addr_v4) => Some(addr_v4),
                        _ => None,
                    })
                    .collect::<Box<[_]>>()
            })
        })
        .flatten()
        .collect()
}

#[cfg(relays)]
fn into_urls<T: reqwest::IntoUrl + Clone>(relays: &[T]) -> Result<Vec<Url>, InvalidRelayUrl> {
    relays
        .iter()
        .map(|url| match url.clone().into_url() {
            Err(e) => Err(InvalidRelayUrl::Parse(e)),
            Ok(url) => {
                if url.scheme() != "http" && url.scheme() != "https" {
                    Err(InvalidRelayUrl::NotHttp(url.to_string()))
                } else {
                    Ok(url)
                }
            }
        })
        .collect::<Result<Vec<Url>, InvalidRelayUrl>>()
}

#[cfg(relays)]
#[derive(thiserror::Error, Debug)]
/// Errors returned when configuring relay URLs for a [`Client`].
pub enum InvalidRelayUrl {
    #[error("Failed to parse into a Url: {0}")]
    /// Failed to parse a URL.
    Parse(reqwest::Error),

    #[error("Relays Urls should have `http` or `https`: {0}")]
    /// Relay URL does not use the `http` or `https` scheme.
    NotHttp(String),
}

#[cfg(all(test, dht))]
mod tests {
    use super::*;

    #[test]
    fn default_dht_timeout_matches_client_timeout() {
        let config = Config::default();
        let dht_config = config
            .dht
            .as_ref()
            .expect("dht should be enabled by default");

        assert_eq!(config.request_timeout, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(dht_config.request_timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn recreated_dht_config_uses_current_client_timeout() {
        let timeout = Duration::from_secs(5);
        let mut builder = Client::builder();

        builder
            .request_timeout(timeout)
            .no_dht()
            .dht(|config| config);

        let dht_config = builder
            .0
            .dht
            .as_ref()
            .expect("dht config should be recreated");

        assert_eq!(dht_config.request_timeout, timeout);
    }

    #[test]
    fn default_dht_report_policy_is_mainnet() {
        let config = Config::default();

        assert_eq!(
            config.dht_report_policy,
            crate::dht::ReportPolicy::mainnet()
        );
    }

    #[test]
    fn custom_dht_report_policy_is_stored() {
        let mut builder = Client::builder();
        builder.dht_report_policy(crate::dht::ReportPolicy::testnet());

        assert_eq!(
            builder.0.dht_report_policy,
            crate::dht::ReportPolicy::testnet()
        );
    }
}

#[cfg(all(test, relays))]
mod relay_tests {
    use super::*;

    #[test]
    fn custom_reqwest_client_is_stored() {
        let mut builder = Client::builder();
        builder.reqwest_client(reqwest::Client::new());

        assert!(builder.0.reqwest_client.is_some());
    }
}
