//! Resolve one canonical client IP per request before either IP rate limiter.

use std::net::{IpAddr, SocketAddr};

use anyhow::{ensure, Context};
use axum::{
    extract::{ConnectInfo, Request, State},
    middleware::map_request_with_state,
    Router,
};
use http::{request, HeaderName, StatusCode};
use ipnet::IpNet;
use tower_governor::{errors::GovernorError, key_extractor::KeyExtractor};

use crate::config::ClientIdentityConfig;

/// Canonical client IP resolved for this request, shared by both rate limiters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RealIp(pub(crate) IpAddr);

/// Extracts a request's normalized real IP from extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealIpKeyExtractor;

impl KeyExtractor for RealIpKeyExtractor {
    type Key = IpAddr;

    fn extract<T>(&self, req: &request::Request<T>) -> Result<Self::Key, GovernorError> {
        req.extensions()
            .get::<RealIp>()
            .map(|real_ip| real_ip.0)
            .ok_or(GovernorError::UnableToExtractKey)
    }
}

/// Runtime policy prepared before opening relay resources.
pub(crate) enum ClientIdentity {
    Direct,
    HttpHeader {
        header: HeaderName,
        trusted_proxies: Vec<IpNet>,
    },
}

impl ClientIdentity {
    pub(crate) fn from_config(config: ClientIdentityConfig) -> anyhow::Result<Self> {
        match config {
            ClientIdentityConfig::Direct {} => Ok(Self::Direct),
            ClientIdentityConfig::HttpHeader {
                header,
                trusted_proxies,
            } => {
                let header = header
                    .parse::<HeaderName>()
                    .context("http.client_identity.header must be a valid HTTP header name")?;
                ensure!(
                    !trusted_proxies.is_empty(),
                    "http.client_identity.trusted_proxies must not be empty"
                );
                Ok(Self::HttpHeader {
                    header,
                    trusted_proxies,
                })
            }
        }
    }

    pub(crate) fn trusts_peer(&self, peer: IpAddr) -> bool {
        let Self::HttpHeader {
            trusted_proxies, ..
        } = self
        else {
            return true;
        };
        // Dual-stack listeners may report an IPv4 peer as mapped IPv6.
        let canonical_peer = peer.to_canonical();
        trusted_proxies
            .iter()
            .any(|network| network.contains(&peer) || network.contains(&canonical_peer))
    }

    /// Resolve identity before rate limiting, CORS, and fallback handlers.
    /// The HTTP server must check the peer allowlist before serving requests.
    pub(crate) fn layer(&self, router: Router) -> Router {
        let header = match self {
            Self::Direct => None,
            Self::HttpHeader { header, .. } => Some(header.clone()),
        };
        router.layer(map_request_with_state(header, resolve_client_ip))
    }
}

async fn resolve_client_ip(
    State(header): State<Option<HeaderName>>,
    mut request: Request,
) -> Result<Request, StatusCode> {
    let ip = extract_client_ip(&request, header.as_ref()).inspect_err(|status| {
        if let Some(header) = &header {
            tracing::debug!(
                %header,
                %status,
                "rejected request: missing, duplicate, or invalid client IP header"
            );
        }
    })?;
    request.extensions_mut().insert(RealIp(ip.to_canonical()));
    Ok(request)
}

fn extract_client_ip(request: &Request, header: Option<&HeaderName>) -> Result<IpAddr, StatusCode> {
    let Some(header) = header else {
        return request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|peer| peer.0.ip())
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR);
    };

    let mut values = request.headers().get_all(header).iter();
    let value = values.next().ok_or(StatusCode::BAD_REQUEST)?;
    if values.next().is_some() {
        return Err(StatusCode::BAD_REQUEST);
    }
    value
        .to_str()
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .parse()
        .map_err(|_| StatusCode::BAD_REQUEST)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_networks_match_original_and_canonical_peer_addresses() {
        for (network, peer, trusted) in [
            ("::ffff:127.0.0.1/128", "::ffff:127.0.0.1", true),
            ("127.0.0.1/32", "::ffff:127.0.0.1", true),
            ("127.0.0.1/32", "127.0.0.1", true),
            ("2001:db8::/64", "2001:db8::1", true),
            ("::ffff:127.0.0.1/128", "::ffff:127.0.0.2", false),
            ("127.0.0.1/32", "::ffff:127.0.0.2", false),
            ("2001:db8::/64", "2001:db8:1::1", false),
        ] {
            let identity = ClientIdentity::from_config(ClientIdentityConfig::HttpHeader {
                header: "X-Real-IP".into(),
                trusted_proxies: vec![network.parse().unwrap()],
            })
            .unwrap();
            assert_eq!(
                identity.trusts_peer(peer.parse().unwrap()),
                trusted,
                "network {network}, peer {peer}"
            );
        }
    }

    #[test]
    fn real_ip_key_extractor_requires_resolved_identity() {
        let mut request = Request::new(axum::body::Body::empty());
        assert!(RealIpKeyExtractor.extract(&request).is_err());
        let ip: IpAddr = "203.0.113.1".parse().unwrap();
        request.extensions_mut().insert(RealIp(ip));
        assert_eq!(RealIpKeyExtractor.extract(&request).unwrap(), ip);
    }
}
