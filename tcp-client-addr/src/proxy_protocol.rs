//! Trust checks and PROXY protocol preface parsing for proxied listeners.

use std::{
    io,
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use ipnet::IpNet;
use ppp::{v1, v2};
use tokio::{io::AsyncReadExt, net::TcpStream, time::timeout};

use crate::identity::{ConfigError, IdentifyError};

const DEFAULT_HEADER_TIMEOUT: Duration = Duration::from_secs(1);
const DEFAULT_MAX_HEADER_BYTES: usize = 4096;
const MIN_MAX_HEADER_BYTES: usize = 28;
const MAX_MAX_HEADER_BYTES: usize = 16 + u16::MAX as usize;
// PROXY v1's maximum complete header length, including CRLF.
const V1_MAX_HEADER_BYTES: usize = 107;
const V2_SIGNATURE: &[u8; 12] = b"\r\n\r\n\0\r\nQUIT\n";

/// Configuration for a mandatory PROXY protocol listener.
#[derive(Clone, Debug)]
pub struct ProxyProtocol {
    trusted_proxies: Arc<[IpNet]>,
    header_timeout: Duration,
    max_header_bytes: usize,
}

impl ProxyProtocol {
    /// Set the trusted immediate peers. An empty list is not allowed.
    ///
    /// A trusted peer may declare any client address; restrict network access to
    /// this listener accordingly. Both PROXY protocol v1 and v2 are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::NoTrustedProxies`] if the list is empty.
    pub fn new(trusted_proxies: impl IntoIterator<Item = IpNet>) -> Result<Self, ConfigError> {
        let trusted_proxies: Arc<[IpNet]> = trusted_proxies.into_iter().collect();
        if trusted_proxies.is_empty() {
            return Err(ConfigError::NoTrustedProxies);
        }
        Ok(Self {
            trusted_proxies,
            header_timeout: DEFAULT_HEADER_TIMEOUT,
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
        })
    }

    /// Set the time allowed to read a complete PROXY preface.
    ///
    /// The default is one second. The clock starts when [`crate::IdentityMode::identify`]
    /// begins reading, not when the TCP connection is accepted.
    ///
    /// # Errors
    ///
    /// The timeout must be nonzero and representable by Tokio's clock.
    pub fn with_header_timeout(mut self, header_timeout: Duration) -> Result<Self, ConfigError> {
        if header_timeout.is_zero()
            || tokio::time::Instant::now()
                .checked_add(header_timeout)
                .is_none()
        {
            return Err(ConfigError::InvalidHeaderTimeout);
        }
        self.header_timeout = header_timeout;
        Ok(self)
    }

    /// Limit the complete PROXY preface length, including v2 TLVs.
    /// PROXY v1 has an additional fixed 107-byte protocol limit.
    ///
    /// # Errors
    ///
    /// The limit must be between 28 and 65,551 bytes, inclusive.
    pub fn with_max_header_bytes(mut self, bytes: usize) -> Result<Self, ConfigError> {
        if !(MIN_MAX_HEADER_BYTES..=MAX_MAX_HEADER_BYTES).contains(&bytes) {
            return Err(ConfigError::InvalidHeaderLimit);
        }
        self.max_header_bytes = bytes;
        Ok(self)
    }

    pub(crate) async fn identify_source(
        &self,
        stream: &mut TcpStream,
        peer: SocketAddr,
    ) -> Result<SocketAddr, IdentifyError> {
        if !self.trusts(peer) {
            return Err(IdentifyError::UntrustedPeer(peer));
        }
        match timeout(
            self.header_timeout,
            read_proxy_source(stream, self.max_header_bytes),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(IdentifyError::HeaderTimeout),
        }
    }

    fn trusts(&self, peer: SocketAddr) -> bool {
        let peer_ip = peer.ip();
        let mapped_ipv4 = match peer_ip {
            IpAddr::V6(ip) => ip.to_ipv4_mapped().map(IpAddr::V4),
            IpAddr::V4(_) => None,
        };
        self.trusted_proxies
            .iter()
            .any(|net| net.contains(&peer_ip) || mapped_ipv4.is_some_and(|ip| net.contains(&ip)))
    }
}

async fn read_proxy_source(
    stream: &mut TcpStream,
    max_bytes: usize,
) -> Result<SocketAddr, IdentifyError> {
    let mut first = [0];
    read_exact(stream, &mut first).await?;

    match first[0] {
        b'P' => read_v1_source(stream, max_bytes).await,
        b'\r' => read_v2_source(stream, max_bytes).await,
        _ => Err(IdentifyError::Malformed),
    }
}

async fn read_v1_source(
    stream: &mut TcpStream,
    max_bytes: usize,
) -> Result<SocketAddr, IdentifyError> {
    let max_bytes = max_bytes.min(V1_MAX_HEADER_BYTES);
    let mut header = Vec::with_capacity(max_bytes);
    header.push(b'P');
    let mut prefix = [0; 5];
    read_exact(stream, &mut prefix).await?;
    if &prefix != b"ROXY " {
        return Err(IdentifyError::Malformed);
    }
    header.extend_from_slice(&prefix);

    loop {
        if header.len() == max_bytes {
            return Err(IdentifyError::HeaderTooLong);
        }
        let mut byte = [0];
        read_exact(stream, &mut byte).await?;
        header.push(byte[0]);
        if header.ends_with(b"\r\n") {
            break;
        }
    }

    let parsed = v1::Header::try_from(header.as_slice()).map_err(|_| IdentifyError::Malformed)?;
    match parsed.addresses {
        v1::Addresses::Tcp4(addresses) => Ok(SocketAddr::new(
            addresses.source_address.into(),
            addresses.source_port,
        )),
        v1::Addresses::Tcp6(addresses) => Ok(SocketAddr::new(
            addresses.source_address.into(),
            addresses.source_port,
        )),
        v1::Addresses::Unknown => Err(IdentifyError::MissingTcpSource),
    }
}

async fn read_v2_source(
    stream: &mut TcpStream,
    max_bytes: usize,
) -> Result<SocketAddr, IdentifyError> {
    let mut fixed = [0; 16];
    fixed[0] = b'\r';
    read_exact(stream, &mut fixed[1..12]).await?;
    if &fixed[..12] != V2_SIGNATURE {
        return Err(IdentifyError::Malformed);
    }
    read_exact(stream, &mut fixed[12..]).await?;

    // Reject unsupported connections before allocating or waiting for a payload.
    // Byte 12 contains the version and command; byte 13 the family and protocol.
    match fixed[12] {
        0x21 => {}                                           // v2 PROXY
        0x20 => return Err(IdentifyError::MissingTcpSource), // v2 LOCAL
        _ => return Err(IdentifyError::Malformed),
    }
    let address_bytes = match fixed[13] {
        0x11 => 12, // IPv4 TCP
        0x21 => 36, // IPv6 TCP
        byte if byte >> 4 > 3 || byte & 0x0f > 2 => {
            return Err(IdentifyError::Malformed);
        }
        _ => return Err(IdentifyError::MissingTcpSource),
    };
    let payload_len = u16::from_be_bytes([fixed[14], fixed[15]]) as usize;
    if payload_len < address_bytes {
        return Err(IdentifyError::Malformed);
    }
    let total_len = 16 + payload_len;
    if total_len > max_bytes {
        return Err(IdentifyError::HeaderTooLong);
    }
    let mut header = vec![0; total_len];
    header[..16].copy_from_slice(&fixed);
    read_exact(stream, &mut header[16..]).await?;
    let parsed = v2::Header::try_from(header.as_slice()).map_err(|_| IdentifyError::Malformed)?;
    // Only addresses are used. Extra payload bytes (TLVs) are consumed but neither
    // interpreted nor validated, including any optional CRC32C checksum.
    match parsed.addresses {
        v2::Addresses::IPv4(addresses) => Ok(SocketAddr::new(
            addresses.source_address.into(),
            addresses.source_port,
        )),
        v2::Addresses::IPv6(addresses) => Ok(SocketAddr::new(
            addresses.source_address.into(),
            addresses.source_port,
        )),
        v2::Addresses::Unspecified | v2::Addresses::Unix(_) => Err(IdentifyError::MissingTcpSource),
    }
}

async fn read_exact(stream: &mut TcpStream, bytes: &mut [u8]) -> Result<(), IdentifyError> {
    match stream.read_exact(bytes).await {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
            Err(IdentifyError::UnexpectedEof)
        }
        Err(error) => Err(IdentifyError::Io(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_ipv6_peer_matches_either_trusted_cidr_form() {
        let peer: SocketAddr = "[::ffff:127.0.0.1]:1234".parse().unwrap();

        for cidr in ["::ffff:127.0.0.1/128", "127.0.0.1/32"] {
            let config = ProxyProtocol::new([cidr.parse().unwrap()]).unwrap();
            assert!(config.trusts(peer), "peer should match {cidr}");
        }

        let config = ProxyProtocol::new(["192.0.2.0/24".parse().unwrap()]).unwrap();
        assert!(!config.trusts(peer));
    }
}
