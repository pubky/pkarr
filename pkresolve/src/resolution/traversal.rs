//! Provider-independent traversal state and completion checks.

use std::{error::Error as StdError, sync::Arc, time::Instant};

use crate::{Error, PublicKey, ServiceName, SignedPacket};

use super::{
    routing::{https_record_name, resolution_from_packet},
    Resolution, ResolutionRequest,
};

/// A traversal either yields ownership of its pending lookup or finishes.
/// Keeping the continuation in `NeedPacket` prevents supplying another packet
/// after completion. Drivers only retrieve packets; transitions own policy.
pub(super) enum ResolutionStep<'a> {
    NeedPacket(ResolutionTraversal<'a>),
    Complete(Resolution),
}

/// Resolution state retained while the driver obtains the routing packet.
/// Only direct routing is supported today. Delegation can add traversal state
/// and return another `NeedPacket` from `accept_packet` without changing drivers.
pub(super) struct ResolutionTraversal<'a> {
    request: &'a ResolutionRequest,
    routing_owner: ServiceName,
}

impl<'a> ResolutionTraversal<'a> {
    pub(super) fn start(request: &'a ResolutionRequest) -> Result<ResolutionStep<'a>, Error> {
        Ok(ResolutionStep::NeedPacket(Self {
            request,
            routing_owner: https_record_name(request)?,
        }))
    }

    pub(super) fn packet_key(&self) -> &PublicKey {
        self.routing_owner.public_key()
    }

    /// Consume the pending lookup and advance, including missing/error handling.
    pub(super) fn accept_packet<E>(
        self,
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
        let resolution = resolution_from_packet(self.request, &self.routing_owner, &packet)?;
        // Check after all lookups. Future delegation must carry the earliest
        // evidence deadline into this result, including earlier packets.
        if resolution.expires_at <= Instant::now() {
            return Err(Error::ExpiredRecord);
        }
        Ok(ResolutionStep::Complete(resolution))
    }
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
