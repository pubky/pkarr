//! Client identity and errors shared by direct and proxied listeners.

use std::{
    io,
    net::{IpAddr, SocketAddr},
};

use thiserror::Error;
use tokio::net::TcpStream;

use crate::proxy_protocol::ProxyProtocol;

/// How a listener determines the client address.
#[derive(Clone, Debug, Default)]
pub enum IdentityMode {
    /// Use the TCP peer address; ignore any HTTP forwarding headers.
    #[default]
    Direct,
    /// Require a PROXY protocol v1 or v2 preface with a TCP client address from a trusted peer.
    /// V1 `UNKNOWN` and v2 `LOCAL` are rejected because they supply no client IP.
    ProxyProtocol(ProxyProtocol),
}

impl IdentityMode {
    /// Identify the client, returning the same stream at the first application byte.
    ///
    /// In proxy mode, reading the complete PROXY preface has a timeout of one
    /// second by default. The caller should still limit concurrent identification
    /// attempts and may apply a deadline measured from connection acceptance.
    /// Dropping this future also drops the owned stream.
    ///
    /// # Errors
    ///
    /// Rejects untrusted peers and missing, timed-out, malformed, oversized, or
    /// unsupported PROXY address headers. V1 `UNKNOWN` and v2 `LOCAL` are unsupported.
    /// V2 metadata after the address block is consumed without validation.
    /// A PROXY listener never falls back to the peer address.
    pub async fn identify(
        &self,
        mut stream: TcpStream,
    ) -> Result<(TcpStream, ClientAddr), IdentifyError> {
        let peer = stream.peer_addr().map_err(IdentifyError::PeerAddress)?;
        let client = match self {
            Self::Direct => peer,
            Self::ProxyProtocol(config) => config.identify_source(&mut stream, peer).await?,
        };
        Ok((stream, ClientAddr { client, peer }))
    }
}

/// The client claimed by an accepted connection and the immediate TCP peer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClientAddr {
    client: SocketAddr,
    peer: SocketAddr,
}

impl ClientAddr {
    /// The client address as received: TCP peer in direct mode, PROXY source in proxy mode.
    pub fn client(self) -> SocketAddr {
        self.client
    }

    /// The client address with IPv4-mapped IPv6 converted to IPv4.
    ///
    /// Use this form when passing a client socket address to code that groups
    /// connections by IP, such as an HTTP rate limiter. Native IPv6 addresses
    /// (including their scope IDs) are left unchanged.
    pub fn normalized_client(self) -> SocketAddr {
        match self.client {
            SocketAddr::V6(address) => match address.ip().to_ipv4_mapped() {
                Some(ipv4) => SocketAddr::new(IpAddr::V4(ipv4), address.port()),
                None => self.client,
            },
            SocketAddr::V4(_) => self.client,
        }
    }

    /// The normalized client IP to use as a per-IP rate-limit key.
    pub fn client_ip(self) -> IpAddr {
        self.normalized_client().ip()
    }

    /// The immediate TCP peer, regardless of mode.
    pub fn peer(self) -> SocketAddr {
        self.peer
    }
}

/// Invalid PROXY listener configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A PROXY listener must have at least one trusted peer network.
    #[error("trusted proxies must not be empty")]
    NoTrustedProxies,
    /// The PROXY preface timeout is zero or cannot be represented.
    #[error("PROXY header timeout must be nonzero and representable")]
    InvalidHeaderTimeout,
    /// The maximum PROXY header size is outside the supported range.
    #[error("PROXY header limit must be 28..=65551")]
    InvalidHeaderLimit,
}

/// Why a connection could not be identified.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IdentifyError {
    /// Obtaining the TCP peer address failed.
    #[error("could not get TCP peer address: {0}")]
    PeerAddress(#[source] io::Error),
    /// The immediate peer is outside all trusted networks.
    #[error("untrusted PROXY peer: {0}")]
    UntrustedPeer(SocketAddr),
    /// The peer did not complete its PROXY preface before the timeout.
    #[error("PROXY preface timed out")]
    HeaderTimeout,
    /// The peer closed the connection before completing its PROXY preface.
    #[error("connection ended during PROXY preface")]
    UnexpectedEof,
    /// The PROXY preface exceeds the configured size limit.
    #[error("PROXY preface exceeds size limit")]
    HeaderTooLong,
    /// The preface is not valid PROXY protocol v1 or v2.
    #[error("malformed PROXY preface")]
    Malformed,
    /// The preface does not declare an IPv4 or IPv6 TCP source address.
    #[error("PROXY preface has no TCP source address")]
    MissingTcpSource,
    /// Reading the connection failed.
    #[error("could not read PROXY preface: {0}")]
    Io(#[source] io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_client_preserves_native_addresses_and_maps_ipv4() {
        let peer: SocketAddr = "127.0.0.1:1234".parse().unwrap();
        let mapped: SocketAddr = "[::ffff:198.51.100.8]:4242".parse().unwrap();
        let native: SocketAddr = "[fe80::1%3]:4242".parse().unwrap();

        let mapped_addr = ClientAddr {
            client: mapped,
            peer,
        };
        assert_eq!(mapped_addr.client(), mapped);
        assert_eq!(
            mapped_addr.normalized_client(),
            "198.51.100.8:4242".parse().unwrap()
        );
        assert_eq!(
            mapped_addr.client_ip(),
            "198.51.100.8".parse::<IpAddr>().unwrap()
        );

        let native_addr = ClientAddr {
            client: native,
            peer,
        };
        assert_eq!(native_addr.normalized_client(), native);
        assert_eq!(native_addr.client_ip(), native.ip());
    }
}
