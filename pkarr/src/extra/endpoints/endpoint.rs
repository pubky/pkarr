use crate::{
    dns::{
        rdata::{RData, SVCB},
        ResourceRecord,
    },
    PublicKey, SignedPacket,
};
use std::{
    collections::HashSet,
    net::{IpAddr, SocketAddr, ToSocketAddrs},
};

use rand::{seq::SliceRandom, thread_rng};

#[derive(Debug, Clone)]
/// An alternative Endpoint for a `qname`, from either [RData::SVCB] or [RData::HTTPS] dns records
pub struct Endpoint {
    target: String,
    public_key: PublicKey,
    port: u16,
    /// SocketAddrs from the [SignedPacket]
    addrs: Vec<IpAddr>,
}

impl Endpoint {
    /// Returns a stack of endpoints from a SignedPacket
    ///
    /// 1. Find the SVCB or HTTPS records
    /// 2. Sort them by priority (reverse)
    /// 3. Shuffle records within each priority
    /// 3. If the target is `.`, keep track of A and AAAA records see [rfc9460](https://www.rfc-editor.org/rfc/rfc9460#name-special-handling-of-in-targ)
    pub(crate) fn parse(
        signed_packet: &SignedPacket,
        target: &str,
        // TODO: change is_svcb to a better name
        is_svcb: bool,
    ) -> Vec<Endpoint> {
        let mut records = signed_packet
            .resource_records(target)
            .filter_map(|record| get_svcb(record, is_svcb))
            .collect::<Vec<_>>();

        // TODO: support wildcard?

        // Shuffle the vector first
        let mut rng = thread_rng();
        records.shuffle(&mut rng);
        // Sort by priority
        records.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut addrs = HashSet::new();
        for record in signed_packet.resource_records("@") {
            match &record.rdata {
                RData::A(ip) => {
                    addrs.insert(IpAddr::V4(ip.address.into()));
                }
                RData::AAAA(ip) => {
                    addrs.insert(IpAddr::V6(ip.address.into()));
                }
                _ => {}
            }
        }
        let addrs = addrs.into_iter().collect::<Vec<_>>();

        records
            .into_iter()
            .map(|s| {
                let target = s.target.to_string();

                let target = if target == "." || target.is_empty() {
                    ".".to_string()
                } else {
                    target
                };

                let port = s
                    .get_param(SVCB::PORT)
                    .map(|bytes| {
                        let mut arr = [0_u8; 2];
                        arr[0] = bytes[0];
                        arr[1] = bytes[1];

                        u16::from_be_bytes(arr)
                    })
                    .unwrap_or_default();

                let addrs = if &target == "." {
                    addrs.clone()
                } else {
                    Vec::with_capacity(0)
                };

                Endpoint {
                    target,
                    port,
                    public_key: signed_packet.public_key(),
                    addrs,
                }
            })
            .collect::<Vec<_>>()
    }

    /// Returns the [SVCB] record's `target` value.
    ///
    /// Useful in web browsers where we can't use [Self::to_socket_addrs]
    pub fn domain(&self) -> &str {
        &self.target
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Return the [PublicKey] of the [SignedPacket] this endpoint was found at.
    ///
    /// This is useful as the [PublicKey] of the endpoint (server), and could be
    /// used for TLS.
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    /// Return an iterator of [SocketAddr], either by resolving the [Endpoint::domain] using normal DNS,
    /// or, if the target is ".", return the [RData::A] or [RData::AAAA] records
    /// from the endpoint's [SignedPacket], if available.
    pub fn to_socket_addrs(&self) -> Vec<SocketAddr> {
        if self.target == "." {
            let port = self.port;

            return self
                .addrs
                .iter()
                .map(|addr| SocketAddr::from((*addr, port)))
                .collect::<Vec<_>>();
        }

        if cfg!(target_arch = "wasm32") {
            vec![]
        } else {
            format!("{}:{}", self.target, self.port)
                .to_socket_addrs()
                .map_or(vec![], |v| v.collect::<Vec<_>>())
        }
    }
}

fn get_svcb<'a>(record: &'a ResourceRecord, is_svcb: bool) -> Option<&'a SVCB<'a>> {
    match &record.rdata {
        RData::SVCB(svcb) => {
            if is_svcb {
                Some(svcb)
            } else {
                None
            }
        }

        RData::HTTPS(curr) => {
            if is_svcb {
                None
            } else {
                Some(&curr.0)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Keypair;

    #[tokio::test]
    async fn endpoint_domain() {
        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .https(
                "foo".try_into().unwrap(),
                SVCB::new(0, "https.example.com".try_into().unwrap()),
                3600,
            )
            .svcb(
                "foo".try_into().unwrap(),
                SVCB::new(0, "protocol.example.com".try_into().unwrap()),
                3600,
            )
            // Make sure SVCB only follows SVCB
            .https(
                "foo".try_into().unwrap(),
                SVCB::new(0, "https.example.com".try_into().unwrap()),
                3600,
            )
            .svcb(
                "_foo".try_into().unwrap(),
                SVCB::new(0, "protocol.example.com".try_into().unwrap()),
                3600,
            )
            .sign(&keypair)
            .unwrap();

        let tld = keypair.public_key();

        // Follow foo.tld HTTPS records
        let endpoint = Endpoint::parse(&signed_packet, &format!("foo.{tld}"), false)
            .pop()
            .unwrap();
        assert_eq!(endpoint.domain(), "https.example.com");

        // Follow _foo.tld SVCB records
        let endpoint = Endpoint::parse(&signed_packet, &format!("_foo.{tld}"), true)
            .pop()
            .unwrap();
        assert_eq!(endpoint.domain(), "protocol.example.com");
    }

    #[test]
    fn endpoint_to_socket_addrs() {
        let mut svcb = SVCB::new(1, ".".try_into().unwrap());
        svcb.set_port(6881);

        let keypair = Keypair::random();
        let signed_packet = SignedPacket::builder()
            .address(
                ".".try_into().unwrap(),
                "209.151.148.15".parse().unwrap(),
                3600,
            )
            .address(
                ".".try_into().unwrap(),
                "2a05:d014:275:6201::64".parse().unwrap(),
                3600,
            )
            .https(".".try_into().unwrap(), svcb, 3600)
            .sign(&keypair)
            .unwrap();

        // Follow foo.tld HTTPS records
        let endpoint = Endpoint::parse(
            &signed_packet,
            &signed_packet.public_key().to_string(),
            false,
        )
        .pop()
        .unwrap();

        assert_eq!(endpoint.domain(), ".");

        let mut addrs = endpoint.to_socket_addrs();
        addrs.sort();

        assert_eq!(
            addrs.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            vec!["209.151.148.15:6881", "[2a05:d014:275:6201::64]:6881"]
        )
    }
}
