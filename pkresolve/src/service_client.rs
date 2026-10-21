//! Cache-first resolution and the non-streaming PKResolve service protocol.
//!
//! Available with the `service-client` feature.
//!
//! Call [`ServiceClient::prepare`], fetch a bundle when needed, then call
//! [`ServiceClient::accept_bundle`]. The client owns protocol handling and caching.
//! Packet validation belongs to PKARR's native `SignedPacket::deserialize()`.
//! The current PKARR decoder still needs upstream fixes before untrusted
//! service responses can be accepted safely.

use std::{collections::HashMap, sync::Arc};

use crate::service_protocol::{ResolveBundle, BUNDLE_SCHEMA};
use crate::{PublicKey, Resolution, ResolutionRequest, SignedPacket};
use base64::{engine::general_purpose::STANDARD, Engine};

/// A response could not be decoded or prove the requested resolution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Invalid JSON or missing response fields.
    #[error("invalid bundle JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A packet is not standard base64.
    #[error("invalid packet encoding: {0}")]
    Base64(#[from] base64::DecodeError),
    /// PKARR could not deserialize a packet.
    #[error("invalid signed packet: {0}")]
    Packet(#[from] pkarr::dns::SimpleDnsError),
    /// The response uses a different protocol revision.
    #[error("unsupported bundle schema: {0}")]
    UnsupportedSchema(u32),
    /// Each key must occur once in a response.
    #[error("duplicate packet for {0}")]
    DuplicatePacket(String),
    /// The bundle is incomplete, expired, or cannot resolve the original request.
    #[error("bundle resolution failed: {0}")]
    Resolution(#[from] crate::Error),
}

/// Result of checking the local cache for an original request.
#[derive(Debug)]
pub enum Preparation {
    /// Fresh endpoints are available without contacting the service.
    Ready(Resolution),
    /// The host must fetch one complete bundle for this request.
    Fetch(BundleRequest),
}

/// A validated original request and its service request body.
/// Keep this value until the HTTP response arrives; it binds bundle acceptance
/// to the original service, URL port, and protocol preferences.
#[derive(Debug)]
pub struct BundleRequest {
    request: ResolutionRequest,
    body: String,
}

impl BundleRequest {
    /// JSON body to POST to the configured HTTPS `/v1/resolve` endpoint.
    pub fn body(&self) -> &str {
        &self.body
    }
}

/// Resolution state with a decoded packet cache. Performs no network I/O.
#[derive(Debug, Default)]
pub struct ServiceClient {
    packets: HashMap<PublicKey, Arc<SignedPacket>>,
}

impl ServiceClient {
    /// Resolve from cache or prepare a bundle fetch. Only missing or expired
    /// packets require HTTP; other resolution errors remain terminal.
    ///
    /// # Errors
    /// Propagates request encoding and terminal resolution failures.
    pub fn prepare(&self, request: ResolutionRequest) -> Result<Preparation, Error> {
        match crate::derive_resolution(&request, &self.packets) {
            Ok(resolution) => Ok(Preparation::Ready(resolution)),
            Err(crate::Error::MissingPacket(_) | crate::Error::ExpiredRecord) => {
                let body = serde_json::to_string(&request)?;
                Ok(Preparation::Fetch(BundleRequest { request, body }))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Resolve a complete bundle for its retained original request, then cache
    /// its decoded packets. Failed responses leave the cache unchanged. Native
    /// `last_seen` values are preserved; incomplete bundles never trigger HTTP.
    /// The required resolution summary is included for convenience. This client
    /// derives its own routes and expiry for the original request, using the
    /// bundled packets as its packet source rather than trusting the summary.
    ///
    /// # Errors
    /// Rejects decoding failures, duplicate keys, and unsuccessful resolution.
    pub fn accept_bundle(
        &mut self,
        bundle_request: BundleRequest,
        response: &[u8],
    ) -> Result<Resolution, Error> {
        let packets = decode_bundle(response)?;
        let resolution = crate::derive_resolution(&bundle_request.request, &packets)?;
        self.packets.extend(packets);
        Ok(resolution)
    }
}

fn decode_bundle(response: &[u8]) -> Result<HashMap<PublicKey, Arc<SignedPacket>>, Error> {
    let bundle: ResolveBundle = serde_json::from_slice(response)?;
    if bundle.schema != BUNDLE_SCHEMA {
        return Err(Error::UnsupportedSchema(bundle.schema));
    }
    let mut packets = HashMap::with_capacity(bundle.packets.len());
    for encoded in bundle.packets {
        let bytes = STANDARD.decode(encoded)?;
        let packet = SignedPacket::deserialize(&bytes)?;
        let key = packet.public_key();
        if packets.insert(key.clone(), Arc::new(packet)).is_some() {
            return Err(Error::DuplicatePacket(key.to_string()));
        }
    }
    Ok(packets)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use pkarr::{dns::rdata::SVCB, Keypair};

    use super::*;
    use crate::HttpProtocol;

    #[test]
    fn expired_cached_routing_failure_requests_a_fresh_bundle() {
        let key = Keypair::from_secret_key(&[1; 32]);
        let mut packet = SignedPacket::builder()
            .https(
                ".".try_into().unwrap(),
                SVCB::new(0, ".".try_into().unwrap()),
                60,
            )
            .sign(&key)
            .unwrap();
        packet.set_last_seen(&(SystemTime::now() - Duration::from_secs(61)).into());
        let client = ServiceClient {
            packets: HashMap::from([(key.public_key(), Arc::new(packet))]),
        };
        let request = ResolutionRequest::new(
            key.to_z32().parse().unwrap(),
            443,
            vec![HttpProtocol::Http11],
        )
        .unwrap();
        assert!(matches!(
            client.prepare(request).unwrap(),
            Preparation::Fetch(_)
        ));
    }
}
