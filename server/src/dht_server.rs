use std::net::SocketAddr;

use pkarr::{
    cache::PkarrCache,
    client::mainline::{
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
};

use tracing::{debug, instrument};

use crate::cache::HeedPkarrCache;

#[derive(Debug)]
/// DhtServer with Rate limiting
pub struct DhtServer {
    inner: mainline::server::DhtServer,
    resolvers: Option<Vec<SocketAddr>>,
    cache: Box<crate::cache::HeedPkarrCache>,
    minimum_ttl: u32,
    maximum_ttl: u32,
}

impl DhtServer {
    pub fn new(
        cache: Box<HeedPkarrCache>,
        resolvers: Option<Vec<SocketAddr>>,
        minimum_ttl: u32,
        maximum_ttl: u32,
    ) -> Self {
        Self {
            // Default DhtServer used to stay a good citizen servicing the Dht.
            inner: mainline::server::DhtServer::default(),
            cache,
            resolvers,
            minimum_ttl,
            maximum_ttl,
        }
    }
}

impl Server for DhtServer {
    #[instrument(skip(self))]
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
            let should_query = if let Some(cached) = self.cache.get(target) {
                // Respond with what we have, even if expired.
                let mutable_item = MutableItem::from(&cached);
                debug!(?target, "cache hit responding with packet!");

                rpc.response(
                    from,
                    transaction_id,
                    ResponseSpecific::GetMutable(GetMutableResponseArguments {
                        responder_id: *rpc.id(),
                        // Token doesn't matter much, as we are most likely _not_ the
                        // closest nodes, so we shouldn't expect an PUT requests based on
                        // this response.
                        token: vec![0, 0, 0, 0],
                        nodes: None,
                        v: mutable_item.value().to_vec(),
                        k: mutable_item.key().to_vec(),
                        seq: *mutable_item.seq(),
                        sig: mutable_item.signature().to_vec(),
                    }),
                );

                // If expired, we try to hydrate the packet from the DHT.
                let expires_in = cached.expires_in(self.minimum_ttl, self.maximum_ttl);
                let expired = expires_in == 0;

                if expired {
                    debug!(
                        ?target,
                        ?expires_in,
                        "cache expired, querying the DHT to hydrate our cache for later."
                    );
                };

                expired
            } else {
                // TODO: rate limit nodes that are making too many request forcing us to making too
                // many queries, either by querying the same non-existent key, or many unique keys.
                debug!(
                    ?target,
                    "cache miss, querying the DHT to hydrate our cache for later."
                );
                true
            };

            if should_query {
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
            }
        };

        // Do normal Dht request handling (peers, mutable, immutable, and routing).
        self.inner
            .handle_request(rpc, from, transaction_id, request)
    }
}
