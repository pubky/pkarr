# Resolvers

> [!WARNING]  
> Deprecated: it is advised to combine a DHT client with an HTTP client to use [Relays](https://pkarr.org/relays) instead.

Resolvers are a special kind of DHT nodes. Like to bootstrapping nodes they are well-known and long running. Unlike normal DHT nodes though, Resolvers act more like DNS resolvers.

## Behavior

Once a resolver receive a `get` request for a mutable value, it checks its big cache of Signed Pkarr packets.

On cache hit, the resolver returns the packet encoded as defined in [Bep_0044](https://www.bittorrent.org/beps/bep_0044.html).

On cache miss, the resolver queries the DHT itself, so subsequent requests to the same Pkarr key can be served from the cache.

## Latency

Since resolvers are hard coded in clients, they offer lower latency than the traversal of the DHT, at least for frequently queried Pkarr keys, on the assumption that Resolvers are using some form of a Least Recently Used (LRU) cache.

## Reliability

Most DHT nodes have limited resources, as they are mostly consumer devices with limited memory and cpu to offer to the network, more over they have high churn rate since consumers don't necessarily care about keeping a high uptime of their nodes.

Resolvers then offer the network more reliable, more generous storage and uptime.

## Censorship resistance

Resolvers should not negatively impact the censorship resistance of Pkarr since they are only used in parallel with the DHT, not instead of it. 

## Relation to relays

Organizations that elect to run a Resolver, may choose to expose the same node and its cache over HTTP using the [Relays](./relays.md) `GET` endpoint.
