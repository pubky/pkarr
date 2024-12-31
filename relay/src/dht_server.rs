use std::{
    fmt::Debug,
    net::{IpAddr, SocketAddrV4},
};

use pkarr::{extra::lmdb_cache::LmdbCache, Cache};

use mainline::{
    rpc::messages::{
        GetMutableResponseArguments, GetValueRequestArguments, MessageType, RequestSpecific,
        RequestTypeSpecific, ResponseSpecific,
    },
    server::{DefaultServer, Server},
    MutableItem, RoutingTable,
};

use tracing::debug;

use crate::rate_limiting::IpRateLimiter;

/// DhtServer with Rate limiting
pub struct DhtServer {
    default_server: DefaultServer,
    resolvers: Option<Box<[SocketAddrV4]>>,
    cache: Box<LmdbCache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
    rate_limiter: Option<IpRateLimiter>,
}

impl Debug for DhtServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resolver")
    }
}

impl DhtServer {
    pub fn new(
        cache: Box<LmdbCache>,
        resolvers: Option<Vec<SocketAddrV4>>,
        minimum_ttl: u32,
        maximum_ttl: u32,
        rate_limiter: Option<IpRateLimiter>,
    ) -> Self {
        Self {
            // Default DhtServer used to stay a good citizen servicing the Dht.
            default_server: DefaultServer::default(),
            cache,
            resolvers: resolvers.map(|r| r.into()),
            minimum_ttl,
            maximum_ttl,
            rate_limiter,
        }
    }
}

impl Server for DhtServer {
    fn handle_request(
        &mut self,
        routing_table: &RoutingTable,
        from: SocketAddrV4,
        request: RequestSpecific,
    ) -> (MessageType, Option<Box<[SocketAddrV4]>>) {
        if let RequestSpecific {
            request_type:
                RequestTypeSpecific::GetValue(GetValueRequestArguments {
                    target,
                    seq,
                    // If there is a salt, we should leave that query to the default server.
                    salt: None,
                }),
            ..
        } = request
        {
            let cached_packet = self.cache.get(target.as_bytes());

            let as_ref = cached_packet.as_ref();

            // Make a background query to the DHT to find and cache packets for subsequent lookups.
            if as_ref
                .as_ref()
                .map(|c| c.is_expired(self.minimum_ttl, self.maximum_ttl))
                .unwrap_or(true)
            {
                // Rate limit nodes that are making too many request forcing us to making too
                // many queries, either by querying the same non-existent key, or many unique keys.
                if self
                    .rate_limiter
                    .clone()
                    .map(|rate_limiter| !rate_limiter.is_limited(&IpAddr::from(*from.ip())))
                    .unwrap_or(true)
                {
                    debug!(?target, "querying the DHT to hydrate our cache for later.");

                    return (
                        MessageType::Request(RequestSpecific {
                            requester_id: *routing_table.id(),
                            request_type: RequestTypeSpecific::GetValue(GetValueRequestArguments {
                                target,
                                seq,
                                salt: None,
                            }),
                        }),
                        self.resolvers.clone(),
                    );
                }

                debug!(?from, "Resolver rate limiting");
            }

            // Respond with what we have, even if expired.
            if let Some(cached_packet) = cached_packet {
                debug!(
                    public_key = ?cached_packet.public_key(),
                    "responding with cached packet even if expired"
                );

                let mutable_item = MutableItem::from(&cached_packet);

                return (
                    MessageType::Response(ResponseSpecific::GetMutable(
                        GetMutableResponseArguments {
                            responder_id: *routing_table.id(),
                            token: self.default_server.tokens.generate_token(from).into(),
                            nodes: Some(routing_table.closest(target)),
                            v: mutable_item.value().into(),
                            k: *mutable_item.key(),
                            seq: mutable_item.seq(),
                            sig: *mutable_item.signature(),
                        },
                    )),
                    None,
                );
            }
        };

        // Do normal Dht request handling (peers, mutable, immutable, and routing).
        self.default_server
            .handle_request(routing_table, from, request)
    }
}
