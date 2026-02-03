use crate::{
    dns::{
        rdata::{RData, SVCParam, SVCB},
        ResourceRecord,
    },
    PublicKey, SignedPacket,
};
use std::{
    collections::{BTreeMap, HashSet},
    net::{IpAddr, SocketAddr, ToSocketAddrs},
};

#[derive(Debug, Clone)]
/// An alternative Endpoint for a `qname`, from either [RData::SVCB] or [RData::HTTPS] dns records
pub struct Endpoint {
    target: String,
    public_key: PublicKey,
    port: u16,
    /// SocketAddrs from the [SignedPacket]
    addrs: Vec<IpAddr>,
    params: BTreeMap<u16, Box<[u8]>>,
}

impl Endpoint {
    /// Returns a stack of endpoints from a SignedPacket
    ///
    /// 1. Find the SVCB or HTTPS records
    /// 2. Sort them by priority (reverse)
    /// 3. Shuffle records within each priority
    /// 3. If the target is `.`, keep track of A and AAAA records see [rfc9460](https://www.rfc-editor.org/rfc/rfc9460#name-special-handling-of-in-targ)
    pub(crate) fn parse(signed_packet: &SignedPacket, target: &str, https: bool) -> Vec<Endpoint> {
        let mut records = signed_packet
            .resource_records(target)
            .filter_map(|record| get_svcb(record, https))
            .collect::<Vec<_>>();

        // Shuffle the vector first
        shuffle(&mut records);
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
                    .iter_params()
                    .find_map(|p| match p {
                        SVCParam::Port(port) => Some(*port),
                        _ => None,
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
                    params: s
                        .iter_params()
                        .map(|(key, value)| (key, value.into()))
                        .collect(),
                }
            })
            .collect::<Vec<_>>()
    }

    // === Getters ===

    /// Returns the [SVCB] record's `target` value.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the [SVCB] record's `target` value, if it is an ICANN domain.
    ///
    /// Returns `None` if the target was `.` or a z32 encoded public key.
    ///
    /// Useful in web browsers where we can't use [Self::to_socket_addrs]
    pub fn domain(&self) -> Option<&str> {
        if self.target != "." && self.target.parse::<PublicKey>().is_err() {
            Some(&self.target)
        } else {
            None
        }
    }

    /// Returns the port number of this endpoint if set to non-zero value.
    pub fn port(&self) -> Option<u16> {
        if self.port > 0 {
            Some(self.port)
        } else {
            None
        }
    }

    /// Return the [PublicKey] of the [SignedPacket] this endpoint was found at.
    ///
    /// This is useful as the [PublicKey] of the endpoint (server), and could be
    /// used for TLS.
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    // === Public Methods ===

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

    /// Returns a service parameter.
    pub fn get_param(&self, key: u16) -> Option<&[u8]> {
        self.params.get(&key).map(|v| v.as_ref())
    }
}

fn get_svcb<'a>(record: &'a ResourceRecord, get_https: bool) -> Option<&'a SVCB<'a>> {
    match &record.rdata {
        RData::SVCB(svcb) => {
            if get_https {
                None
            } else {
                Some(svcb)
            }
        }

        RData::HTTPS(curr) => {
            if get_https {
                Some(&curr.0)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Shuffles a slice randomly.
fn shuffle<T>(slice: &mut [T]) {
    if slice.len() <= 1 {
        return;
    }

    let mut chunk = 0;
    let mut chunk_remaining: u32 = 0;

    for i in 1..slice.len() {
        if chunk_remaining == 0 {
            let mut buf = [0u8; 8];
            getrandom::fill(&mut buf).expect("getrandom failed");
            chunk = u64::from_le_bytes(buf);
            chunk_remaining = 64;
        }

        let j = i + 1;

        let rand_pos = (chunk % j as u64) as usize;
        chunk /= j as u64;

        let bits_used = j.next_power_of_two().trailing_zeros();
        chunk_remaining = chunk_remaining.saturating_sub(bits_used);

        slice.swap(i, rand_pos);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
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
        let endpoint = Endpoint::parse(&signed_packet, &format!("foo.{tld}"), true)
            .pop()
            .unwrap();
        assert_eq!(endpoint.domain(), Some("https.example.com"));

        // Follow _foo.tld SVCB records
        let endpoint = Endpoint::parse(&signed_packet, &format!("_foo.{tld}"), false)
            .pop()
            .unwrap();
        assert_eq!(endpoint.domain(), Some("protocol.example.com"));
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
            true,
        )
        .pop()
        .unwrap();

        assert_eq!(endpoint.target(), ".");
        assert_eq!(endpoint.domain(), None);

        let mut addrs = endpoint.to_socket_addrs();
        addrs.sort();

        assert_eq!(
            addrs.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            vec!["209.151.148.15:6881", "[2a05:d014:275:6201::64]:6881"]
        )
    }
}
