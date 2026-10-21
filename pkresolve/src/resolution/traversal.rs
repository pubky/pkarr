//! Provider-independent traversal state and completion checks.

use std::{
    collections::HashSet,
    error::Error as StdError,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{Error, PublicKey, ServiceName, SignedPacket};

use super::{
    routing::{https_record_name, route_packet, PacketRouting},
    Resolution, ResolutionRequest,
};

const MAX_ROUTING_NAMES: usize = 32;

/// A traversal either yields ownership of its pending lookup or finishes.
/// Keeping the continuation in `NeedPacket` prevents supplying another packet
/// after completion. Drivers only retrieve packets; transitions own policy.
pub(super) enum ResolutionStep<'a> {
    NeedPacket(ResolutionTraversal<'a>),
    Complete(Resolution),
}

/// Resolution state retained while the driver obtains the routing packet.
/// Alias hops retain the original request and the earliest evidence deadline.
pub(super) struct ResolutionTraversal<'a> {
    request: &'a ResolutionRequest,
    routing_owner: ServiceName,
    visited: HashSet<ServiceName>,
    expires_at: Option<Instant>,
}

impl<'a> ResolutionTraversal<'a> {
    pub(super) fn start(request: &'a ResolutionRequest) -> Result<ResolutionStep<'a>, Error> {
        let routing_owner = https_record_name(request)?;
        Ok(ResolutionStep::NeedPacket(Self {
            request,
            visited: HashSet::from([routing_owner.clone()]),
            routing_owner,
            expires_at: None,
        }))
    }

    pub(super) fn packet_key(&self) -> &PublicKey {
        self.routing_owner.public_key()
    }

    /// Consume the pending lookup and advance, including missing/error handling.
    pub(super) fn accept_packet<E>(
        mut self,
        packet: Result<Option<Arc<SignedPacket>>, E>,
    ) -> Result<ResolutionStep<'a>, Error>
    where
        E: StdError + Send + Sync + 'static,
    {
        let key = self.packet_key();
        let packet = packet
            .map_err(|source| Error::Provider {
                key: Box::new(key.clone()),
                source: Box::new(source),
            })?
            .ok_or_else(|| Error::MissingPacket(Box::new(key.clone())))?;
        // Expired evidence cannot authorize either routes or terminal errors.
        // Check before interpreting records so cached failures can be refreshed.
        let packet_deadline = packet_expires_at(&packet)?;
        let expires_at = self
            .expires_at
            .map_or(packet_deadline, |previous| previous.min(packet_deadline));
        if expires_at <= Instant::now() {
            return Err(Error::ExpiredRecord);
        }
        match route_packet(self.request, &self.routing_owner, &packet)? {
            PacketRouting::Alias(target) => {
                if self.visited.contains(&target) {
                    return Err(Error::InvalidRecord("HTTPS alias loop"));
                }
                // Bound names as well as packet owners: aliases within one
                // packet can otherwise bypass the relay's packet budget.
                if self.visited.len() >= MAX_ROUTING_NAMES {
                    return Err(Error::InvalidRecord("HTTPS alias chain exceeds 32 names"));
                }
                self.visited.insert(target.clone());
                self.routing_owner = target;
                self.expires_at = Some(expires_at);
                Ok(ResolutionStep::NeedPacket(self))
            }
            PacketRouting::Endpoints(candidates) => {
                // Record processing also counts against the evidence deadline.
                if expires_at <= Instant::now() {
                    return Err(Error::ExpiredRecord);
                }
                Ok(ResolutionStep::Complete(Resolution {
                    service: self.request.service().clone(),
                    url_port: self.request.url_port(),
                    candidates,
                    expires_at,
                }))
            }
        }
    }
}

/// Convert the shortest record TTL into a monotonic deadline, preserving age.
fn packet_expires_at(packet: &SignedPacket) -> Result<Instant, Error> {
    let ttl = packet.ttl(0, u32::MAX);
    if ttl == 0 {
        return Err(Error::Unsupported("zero TTL"));
    }
    let now = Instant::now();
    // PKARR 8.1's expires_in() underflows for future observations. Treat clock
    // skew as age zero, without extending reuse beyond the packet's TTL.
    // Keep subsecond precision: rounding the age down would extend the TTL.
    let observed_age = Duration::from_micros(
        pkarr::Timestamp::now()
            .as_u64()
            .saturating_sub(packet.last_seen().as_u64()),
    );
    let remaining = Duration::from_secs(u64::from(ttl)).saturating_sub(observed_age);
    now.checked_add(remaining)
        .ok_or(Error::InvalidTime("cache deadline overflow"))
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, net::IpAddr};

    use pkarr::{dns::rdata::SVCB, Keypair};

    use crate::HttpProtocol;

    use super::*;

    #[test]
    fn traversal_preserves_service_scope_and_port_across_packet_lookup() {
        let keypair = Keypair::from_secret_key(&[1; 32]);
        let request = ResolutionRequest::new(
            format!("app.{}", keypair.to_z32()).parse().unwrap(),
            8443,
            vec![HttpProtocol::Http11],
        )
        .unwrap();
        let packet = SignedPacket::builder()
            .https(
                "_8443._https.app".try_into().unwrap(),
                SVCB::new(1, ".".try_into().unwrap()),
                300,
            )
            .address(
                "_8443._https.app".try_into().unwrap(),
                "203.0.113.10".parse().unwrap(),
                300,
            )
            .sign(&keypair)
            .unwrap();

        let ResolutionStep::NeedPacket(traversal) = ResolutionTraversal::start(&request).unwrap()
        else {
            panic!("expected a routing packet request");
        };
        assert_eq!(traversal.packet_key(), &keypair.public_key());

        let step = traversal
            .accept_packet(Ok::<_, Infallible>(Some(Arc::new(packet))))
            .unwrap();
        let ResolutionStep::Complete(resolution) = step else {
            panic!("direct routing should complete after one packet");
        };
        assert_eq!(&resolution.service, request.service());
        assert_eq!(resolution.url_port, 8443);
        assert_eq!(resolution.candidates.len(), 1);
        assert_eq!(resolution.candidates[0].port.get(), 8443);
        assert_eq!(
            resolution.candidates[0].addresses,
            vec!["203.0.113.10".parse::<IpAddr>().unwrap()]
        );
    }
}
