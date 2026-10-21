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
        public_key: *resolution.service.public_key().as_bytes(),
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
