//! Synchronous CXX operations for Chromium-owned HTTP requests.
//! Rust owns packet caching, request encoding, and bundle interpretation.

use std::time::Instant;

use pkresolve::{
    Error, ResolutionRequest, ServiceClient,
    service_client::{Error as ServiceError, Preparation as ServicePreparation},
};

#[cxx::bridge(namespace = "pkresolve")]
pub mod ffi {
    enum HttpProtocol {
        Http11,
        Http2,
    }

    #[derive(PartialEq, Eq)]
    enum StatusCode {
        Ok,
        InvalidInput,
        InvalidBundle,
        Unsupported,
        ServiceUnavailable,
        NoCompatibleEndpoint,
        NoAddresses,
    }

    struct Status {
        code: StatusCode,
        message: String,
    }

    struct EndpointCandidate {
        addresses: Vec<String>,
        port: u16,
        priority: u16,
        protocols: Vec<HttpProtocol>,
    }

    /// Valid only when the response status is `Ok`.
    #[derive(Default)]
    struct Resolution {
        service: String,
        url_port: u16,
        /// Required raw TLS key after HTTPS AliasMode delegation.
        public_key: [u8; 32],
        valid_for_ms: u64,
        candidates: Vec<EndpointCandidate>,
    }

    struct ResolveResponse {
        status: Status,
        resolution: Resolution,
    }

    extern "Rust" {
        type Client;
        type Preparation;

        fn new_client() -> Box<Client>;
        fn prepare(
            self: &Client,
            service: &str,
            url_port: u16,
            protocols: &[HttpProtocol],
        ) -> Box<Preparation>;
        fn needs_bundle(self: &Preparation) -> bool;
        fn request_body(self: &Preparation) -> &str;
        fn prepared_result(preparation: Box<Preparation>) -> ResolveResponse;
        fn accept_bundle(
            self: &mut Client,
            preparation: Box<Preparation>,
            response: &[u8],
        ) -> ResolveResponse;
    }
}

/// Packet cache and protocol operations for one Chromium sequence.
#[derive(Default)]
pub struct Client {
    inner: ServiceClient,
}

/// Create empty resolution state. No HTTP client or runtime is needed.
pub fn new_client() -> Box<Client> {
    Box::new(Client::default())
}

/// Cached result, error, or retained original request for an HTTP fetch.
pub struct Preparation {
    result: Result<ServicePreparation, ServiceError>,
}

impl Preparation {
    /// Whether Chromium must POST the service request body.
    pub fn needs_bundle(&self) -> bool {
        matches!(self.result, Ok(ServicePreparation::Fetch(_)))
    }

    /// Borrow the JSON body for a fetch; copy it before asynchronous HTTP.
    pub fn request_body(&self) -> &str {
        match &self.result {
            Ok(ServicePreparation::Fetch(bundle_request)) => bundle_request.body(),
            _ => "",
        }
    }
}

impl Client {
    /// Check the cache and retain the validated request if HTTP is necessary.
    pub fn prepare(
        &self,
        service: &str,
        url_port: u16,
        protocols: &[ffi::HttpProtocol],
    ) -> Box<Preparation> {
        Box::new(Preparation {
            result: parse_request(service, url_port, protocols)
                .map_err(ServiceError::from)
                .and_then(|request| self.inner.prepare(request)),
        })
    }

    /// Consume the original fetch preparation and the complete HTTP response.
    /// The cache is updated only after the bundle resolves successfully.
    pub fn accept_bundle(
        &mut self,
        preparation: Box<Preparation>,
        response: &[u8],
    ) -> ffi::ResolveResponse {
        match preparation.result {
            Ok(ServicePreparation::Fetch(bundle_request)) => {
                into_response(self.inner.accept_bundle(bundle_request, response))
            }
            _ => failure(
                ffi::StatusCode::InvalidInput,
                "preparation does not require a bundle".into(),
            ),
        }
    }
}

/// Consume a cache hit or terminal error without performing HTTP.
pub fn prepared_result(preparation: Box<Preparation>) -> ffi::ResolveResponse {
    match preparation.result {
        Ok(ServicePreparation::Ready(resolution)) => into_response(Ok(resolution)),
        Err(error) => into_response(Err(error)),
        Ok(ServicePreparation::Fetch(_)) => {
            failure(ffi::StatusCode::InvalidInput, "bundle required".into())
        }
    }
}

fn into_response(result: Result<pkresolve::Resolution, ServiceError>) -> ffi::ResolveResponse {
    match result {
        Ok(resolution) => ffi::ResolveResponse {
            status: ffi::Status {
                code: ffi::StatusCode::Ok,
                message: String::new(),
            },
            resolution: convert_resolution(resolution),
        },
        Err(error) => error_response(error),
    }
}

fn parse_request(
    service: &str,
    url_port: u16,
    protocols: &[ffi::HttpProtocol],
) -> Result<ResolutionRequest, Error> {
    let protocols = protocols
        .iter()
        .map(|protocol| match *protocol {
            ffi::HttpProtocol::Http11 => Ok(pkresolve::HttpProtocol::Http11),
            ffi::HttpProtocol::Http2 => Ok(pkresolve::HttpProtocol::Http2),
            _ => Err(Error::InvalidRequest("unknown HTTP protocol")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    ResolutionRequest::new(service.parse()?, url_port, protocols)
}

fn failure(code: ffi::StatusCode, message: String) -> ffi::ResolveResponse {
    ffi::ResolveResponse {
        status: ffi::Status { code, message },
        resolution: ffi::Resolution::default(),
    }
}

fn error_response(error: ServiceError) -> ffi::ResolveResponse {
    let code = match &error {
        ServiceError::Resolution(Error::InvalidServiceName(_) | Error::InvalidRequest(_)) => {
            ffi::StatusCode::InvalidInput
        }
        ServiceError::Resolution(Error::Unsupported(_)) => ffi::StatusCode::Unsupported,
        ServiceError::Resolution(Error::ServiceUnavailable) => ffi::StatusCode::ServiceUnavailable,
        ServiceError::Resolution(Error::NoCompatibleEndpoint) => {
            ffi::StatusCode::NoCompatibleEndpoint
        }
        ServiceError::Resolution(Error::NoAddresses) => ffi::StatusCode::NoAddresses,
        _ => ffi::StatusCode::InvalidBundle,
    };
    failure(code, error.to_string())
}

fn convert_resolution(resolution: pkresolve::Resolution) -> ffi::Resolution {
    ffi::Resolution {
        service: resolution.service.to_string(),
        url_port: resolution.url_port,
        public_key: *resolution.tls_public_key.as_bytes(),
        valid_for_ms: resolution
            .expires_at
            .saturating_duration_since(Instant::now())
            .as_millis() as u64,
        candidates: resolution
            .candidates
            .into_iter()
            .map(convert_candidate)
            .collect(),
    }
}

fn convert_candidate(candidate: pkresolve::EndpointCandidate) -> ffi::EndpointCandidate {
    ffi::EndpointCandidate {
        addresses: candidate
            .addresses
            .into_iter()
            .map(|ip| ip.to_string())
            .collect(),
        port: candidate.port.get(),
        priority: candidate.priority,
        protocols: candidate
            .protocols
            .into_iter()
            .map(|protocol| match protocol {
                pkresolve::HttpProtocol::Http11 => ffi::HttpProtocol::Http11,
                pkresolve::HttpProtocol::Http2 => ffi::HttpProtocol::Http2,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::HashMap,
        sync::Arc,
        time::{Duration, SystemTime},
    };

    use base64::{Engine, engine::general_purpose::STANDARD};
    use pkarr::{SignedPacket, dns::Name, dns::rdata::SVCB};
    use pkresolve::service_protocol::{BUNDLE_SCHEMA, ResolveBundle};

    #[test]
    fn alias_bundle_derives_routes_and_tls_key_preserves_age_and_reuses_cache() {
        let key = pkarr::Keypair::from_secret_key(&[3; 32]);
        let delegate = pkarr::Keypair::from_secret_key(&[5; 32]);
        let service = format!("app.{}", key.to_z32());
        let target = delegate.to_z32();
        let mut alias = SignedPacket::builder()
            .https(
                Name::new_unchecked("_8443._https.app"),
                SVCB::new(0, Name::new_unchecked(&target)),
                60,
            )
            .sign(&key)
            .unwrap();
        // The alias is the limiting packet, and receipt must not refresh its age.
        alias.set_last_seen(&(SystemTime::now() - Duration::from_secs(20)).into());
        let root = Name::new_unchecked(".");
        let mut route = SVCB::new(1, root.clone());
        route.set_port(9443);
        let packet = SignedPacket::builder()
            .https(root.clone(), route, 120)
            .address(root, "203.0.113.10".parse().unwrap(), 120)
            .sign(&delegate)
            .unwrap();
        let mut client = new_client();
        let preparation = client.prepare(&service, 8443, &[ffi::HttpProtocol::Http11]);
        assert!(preparation.needs_bundle());
        let request: ResolutionRequest = serde_json::from_str(preparation.request_body()).unwrap();
        assert_eq!(request.service().to_string(), service);
        assert_eq!(request.url_port(), 8443);
        assert_eq!(
            request.supported_protocols(),
            [pkresolve::HttpProtocol::Http11]
        );
        // Bundle order is irrelevant; the delegate can precede the origin.
        let packets = vec![
            STANDARD.encode(packet.serialize()),
            STANDARD.encode(alias.serialize()),
        ];
        let provider = HashMap::from([
            (key.public_key(), Arc::new(alias)),
            (delegate.public_key(), Arc::new(packet)),
        ]);
        let mut bundle = ResolveBundle {
            schema: BUNDLE_SCHEMA,
            packets,
            resolution: pkresolve::derive_resolution(&request, &provider)
                .unwrap()
                .into(),
        };
        // The summary is required but is not authoritative for any result field.
        bundle.resolution.service = "untrusted.example".into();
        bundle.resolution.url_port = 80;
        bundle.resolution.tls_public_key = key.to_z32();
        bundle.resolution.candidates.clear();
        bundle.resolution.valid_until_unix_ms = u64::MAX;
        let response = client.accept_bundle(preparation, &serde_json::to_vec(&bundle).unwrap());
        assert!(
            response.status.code == ffi::StatusCode::Ok,
            "{}",
            response.status.message
        );
        let resolution = response.resolution;
        assert_eq!(resolution.service, service);
        assert_eq!(resolution.url_port, 8443);
        assert_eq!(resolution.public_key, *delegate.public_key().as_bytes());
        assert!(resolution.valid_for_ms > 0 && resolution.valid_for_ms <= 40_000);
        assert_eq!(resolution.candidates.len(), 1);
        let candidate = &resolution.candidates[0];
        assert_eq!(candidate.addresses, ["203.0.113.10"]);
        assert_eq!(candidate.port, 9443);
        assert_eq!(candidate.priority, 1);
        assert_eq!(candidate.protocols.len(), 1);
        assert!(candidate.protocols[0] == ffi::HttpProtocol::Http11);

        let cached = client.prepare(&service, 8443, &[ffi::HttpProtocol::Http11]);
        assert!(!cached.needs_bundle());
        assert!(cached.request_body().is_empty());
        let cached = prepared_result(cached);
        assert!(cached.status.code == ffi::StatusCode::Ok);
        assert_eq!(
            cached.resolution.candidates[0].addresses,
            candidate.addresses
        );
        assert_eq!(cached.resolution.service, service);
        assert_eq!(cached.resolution.url_port, 8443);
        assert_eq!(
            cached.resolution.public_key,
            *delegate.public_key().as_bytes()
        );
        assert_eq!(cached.resolution.candidates[0].port, 9443);
        assert!(cached.resolution.valid_for_ms <= resolution.valid_for_ms);
    }

    #[test]
    fn malformed_bundle_returns_error_without_populating_cache() {
        let service = pkarr::Keypair::from_secret_key(&[4; 32]).to_z32();
        let mut client = new_client();
        let preparation = client.prepare(&service, 443, &[ffi::HttpProtocol::Http11]);
        let response = client.accept_bundle(preparation, b"not JSON");
        assert!(response.status.code == ffi::StatusCode::InvalidBundle);
        assert!(!response.status.message.is_empty());
        assert!(response.resolution.candidates.is_empty());
        assert!(
            client
                .prepare(&service, 443, &[ffi::HttpProtocol::Http11])
                .needs_bundle()
        );
    }

    #[test]
    fn conversion_exports_delegated_tls_key_and_original_service() {
        let origin = pkarr::Keypair::from_secret_key(&[1; 32]);
        let delegate = pkarr::Keypair::from_secret_key(&[2; 32]);
        let resolution = pkresolve::Resolution {
            service: origin.to_z32().parse().unwrap(),
            tls_public_key: delegate.public_key(),
            url_port: 8443,
            candidates: Vec::new(),
            expires_at: Instant::now(),
        };

        let converted = convert_resolution(resolution);
        assert_eq!(converted.service, origin.to_z32());
        assert_eq!(converted.public_key, *delegate.public_key().as_bytes());
        assert_eq!(converted.url_port, 8443);
    }
}
