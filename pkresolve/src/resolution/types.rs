//! Requests, endpoint candidates, and resolution results.

use std::{net::IpAddr, num::NonZeroU16, time::Instant};

use crate::{Error, ServiceName};

/// Supported TCP HTTP application protocols. HTTP/3 needs a separate policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "service-protocol",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum HttpProtocol {
    /// HTTP/1.1; the default HTTPS protocol when defaults are allowed.
    #[cfg_attr(feature = "service-protocol", serde(rename = "http/1.1"))]
    Http11,
    /// HTTP/2 over TLS.
    #[cfg_attr(feature = "service-protocol", serde(rename = "h2"))]
    Http2,
}

impl HttpProtocol {
    /// TLS ALPN identifier.
    pub fn alpn(self) -> &'static str {
        match self {
            Self::Http11 => "http/1.1",
            Self::Http2 => "h2",
        }
    }
}

/// Original service scope and client-supported protocols.
/// Raw-key Pubky authentication is the only supported authentication mode.
/// With `service-protocol`, supports Serde using `service`, `url_port`, and
/// `supported_alpn` fields. Deserialization validates all fields and removes
/// duplicate ALPNs while preserving preference order.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "service-protocol",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "service-protocol", serde(deny_unknown_fields))]
pub struct ResolutionRequest {
    service: ServiceName,
    url_port: NonZeroU16,
    #[cfg_attr(
        feature = "service-protocol",
        serde(rename = "supported_alpn", deserialize_with = "deserialize_protocols")
    )]
    supported_protocols: Vec<HttpProtocol>,
}

#[cfg(feature = "service-protocol")]
fn deserialize_protocols<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<HttpProtocol>, D::Error> {
    use serde::{de::Error as _, Deserialize};

    let protocols = Vec::<HttpProtocol>::deserialize(deserializer)?;
    normalize_protocols(protocols).map_err(D::Error::custom)
}

fn normalize_protocols(protocols: Vec<HttpProtocol>) -> Result<Vec<HttpProtocol>, Error> {
    if protocols.is_empty() {
        return Err(Error::InvalidRequest(
            "at least one HTTP protocol is required",
        ));
    }
    let mut unique = Vec::new();
    for protocol in protocols {
        if !unique.contains(&protocol) {
            unique.push(protocol);
        }
    }
    Ok(unique)
}

impl ResolutionRequest {
    /// Construct a request preserving the original port and full service name.
    /// Remove duplicate protocols while preserving client preference order.
    ///
    /// # Errors
    /// Rejects port zero and an empty protocol list.
    pub fn new(
        service: ServiceName,
        url_port: u16,
        supported_protocols: Vec<HttpProtocol>,
    ) -> Result<Self, Error> {
        let url_port =
            NonZeroU16::new(url_port).ok_or(Error::InvalidRequest("URL port must be nonzero"))?;
        let supported_protocols = normalize_protocols(supported_protocols)?;
        Ok(Self {
            service,
            url_port,
            supported_protocols,
        })
    }

    /// Original scoped service name, also retained as SNI and HTTP host.
    pub fn service(&self) -> &ServiceName {
        &self.service
    }

    /// Original URL port, independent of a candidate's connection port.
    pub fn url_port(&self) -> u16 {
        self.url_port.get()
    }

    /// Preserve the validated port when constructing endpoint candidates.
    pub(super) fn nonzero_url_port(&self) -> NonZeroU16 {
        self.url_port
    }

    /// Protocols in client preference order.
    pub fn supported_protocols(&self) -> &[HttpProtocol] {
        &self.supported_protocols
    }
}

/// One compatible route, authenticated with the destination Pubky key.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "service-protocol",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct EndpointCandidate {
    /// Signed addresses and hints, deduplicated. The host chooses address order.
    pub addresses: Vec<IpAddr>,
    /// Selected connection port; it does not change the URL or HTTP authority.
    pub port: NonZeroU16,
    /// Lower values are preferred; callers shuffle equal-priority candidates.
    pub priority: u16,
    /// Compatible ALPNs in client preference order.
    pub protocols: Vec<HttpProtocol>,
}

/// Routes sharing the destination service's raw TLS key and a reuse deadline.
/// This describes how to authenticate a peer; it does not establish TLS identity.
#[derive(Clone, Debug)]
pub struct Resolution {
    /// Original service for HTTP authority and TLS SNI. Every candidate must
    /// authenticate with exactly `service.public_key()` as its raw TLS key;
    /// CA authentication cannot substitute for it.
    pub service: ServiceName,
    /// Original URL port for HTTP authority construction.
    pub url_port: u16,
    /// Candidates sorted by ascending priority. Equal-priority order has no
    /// preference meaning; the host scheduler chooses attempt order.
    pub candidates: Vec<EndpointCandidate>,
    /// Expiry of the packet used to derive these candidates.
    /// The complete result may be reused until this time.
    pub expires_at: Instant,
}

#[cfg(test)]
mod tests {
    use pkarr::Keypair;

    use super::*;

    #[test]
    fn request_preserves_preference_order_without_duplicate_protocols() {
        let service = Keypair::from_secret_key(&[1; 32]).to_z32().parse().unwrap();
        let request = ResolutionRequest::new(
            service,
            443,
            vec![
                HttpProtocol::Http11,
                HttpProtocol::Http2,
                HttpProtocol::Http11,
            ],
        )
        .unwrap();
        assert_eq!(
            request.supported_protocols(),
            &[HttpProtocol::Http11, HttpProtocol::Http2]
        );
    }

    #[test]
    fn request_rejects_zero_port_and_empty_protocols() {
        let service: ServiceName = Keypair::from_secret_key(&[1; 32]).to_z32().parse().unwrap();
        assert!(matches!(
            ResolutionRequest::new(service.clone(), 0, vec![HttpProtocol::Http11]),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            ResolutionRequest::new(service, 443, vec![]),
            Err(Error::InvalidRequest(_))
        ));
    }
}
