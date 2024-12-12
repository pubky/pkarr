use std::{
    fmt::Debug,
    net::{SocketAddr, ToSocketAddrs},
};

use pkarr::{
    extra::lmdb_cache::LmdbCache,
    mainline::{
        self,
        rpc::{
            messages::{
                GetMutableResponseArguments, GetValueRequestArguments, RequestSpecific,
                RequestTypeSpecific, ResponseSpecific,
            },
            Rpc,
        },
        server::Server,
        MutableItem,
    },
    Cache,
};

use tracing::debug;

use crate::rate_limiting::IpRateLimiter;

/// DhtServer with Rate limiting
pub struct DhtServer {
    inner: mainline::server::DefaultServer,
    resolvers: Option<Vec<SocketAddr>>,
    cache: Box<LmdbCache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
    rate_limiter: IpRateLimiter,
}

impl Debug for DhtServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resolver")
    }
}

impl DhtServer {
    pub fn new(
        cache: Box<LmdbCache>,
        resolvers: Option<Vec<SocketAddr>>,
        minimum_ttl: u32,
        maximum_ttl: u32,
        rate_limiter: IpRateLimiter,
    ) -> Self {
        Self {
            // Default DhtServer used to stay a good citizen servicing the Dht.
            inner: mainline::server::DefaultServer::default(),
            cache,
            resolvers: resolvers.map(|resolvers| {
                resolvers
                    .iter()
                    .flat_map(|resolver| resolver.to_socket_addrs())
                    .flatten()
                    .collect::<Vec<_>>()
            }),
            minimum_ttl,
            maximum_ttl,
            rate_limiter,
        }
    }
}

impl Server for DhtServer {
    fn handle_request(
        &mut self,
        rpc: &mut Rpc,
        from: SocketAddr,
        transaction_id: u16,
        request: &RequestSpecific,
    ) {
        if let RequestSpecific {
            request_type: RequestTypeSpecific::GetValue(GetValueRequestArguments { target, .. }),
            ..
        } = request
        {
            let cached_packet = self.cache.get(target.as_bytes());

            let as_ref = cached_packet.as_ref();

            // Should query?
            if as_ref
                .as_ref()
                .map(|c| c.is_expired(self.minimum_ttl, self.maximum_ttl))
                .unwrap_or(true)
            {
                debug!(?target, "querying the DHT to hydrate our cache for later.");

                // Rate limit nodes that are making too many request forcing us to making too
                // many queries, either by querying the same non-existent key, or many unique keys.
                if self.rate_limiter.is_limited(&from.ip()) {
                    debug!(?from, "Resolver rate limiting");
                } else {
                    rpc.get(
                        *target,
                        RequestTypeSpecific::GetValue(GetValueRequestArguments {
                            target: *target,
                            seq: None,
                            salt: None,
                        }),
                        None,
                        self.resolvers.to_owned(),
                    );
                };
            }

            // Respond with what we have, even if expired.
            if let Some(cached_packet) = cached_packet {
                debug!(
                    public_key = ?cached_packet.public_key(),
                    "responding with cached packet even if expired"
                );

                let mutable_item = MutableItem::from(&cached_packet);

                rpc.response(
                    from,
                    transaction_id,
                    ResponseSpecific::GetMutable(GetMutableResponseArguments {
                        responder_id: *rpc.id(),
                        // Token doesn't matter much, as we are most likely _not_ the
                        // closest nodes, so we shouldn't expect a PUT requests based on
                        // this response.
                        token: vec![0, 0, 0, 0],
                        nodes: None,
                        v: mutable_item.value().to_vec(),
                        k: mutable_item.key().to_vec(),
                        seq: *mutable_item.seq(),
                        sig: mutable_item.signature().to_vec(),
                    }),
                );
            }
        };

        // Do normal Dht request handling (peers, mutable, immutable, and routing).
        self.inner
            .handle_request(rpc, from, transaction_id, request)
    }
}
