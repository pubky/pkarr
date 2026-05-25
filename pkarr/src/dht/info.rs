use std::net::SocketAddrV4;

/// Information about the DHT node.
#[derive(Clone, Copy, Debug)]
pub struct DhtInfo {
    local_addr: SocketAddrV4,
    public_address: Option<SocketAddrV4>,
    firewalled: bool,
    dht_size_estimate: (usize, f64),
}

impl DhtInfo {
    pub(crate) fn new(
        local_addr: SocketAddrV4,
        public_address: Option<SocketAddrV4>,
        firewalled: bool,
        dht_size_estimate: (usize, f64),
    ) -> Self {
        Self {
            local_addr,
            public_address,
            firewalled,
            dht_size_estimate,
        }
    }

    /// Local UDP IPv4 socket address that this node is listening on.
    pub fn local_addr(&self) -> SocketAddrV4 {
        self.local_addr
    }

    /// Returns the best guess for this node's public address.
    pub fn public_address(&self) -> Option<SocketAddrV4> {
        self.public_address
    }

    /// Returns true if this node is likely firewalled.
    pub fn firewalled(&self) -> bool {
        self.firewalled
    }

    /// Returns the DHT size estimate and confidence value.
    pub fn dht_size_estimate(&self) -> (usize, f64) {
        self.dht_size_estimate
    }
}
