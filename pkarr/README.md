# Pkarr

Rust implementation of [Pkarr](https://github.com/pubky/pkarr) for signing,
publishing, and resolving DNS packets over
[Mainline DHT](https://github.com/Pubky/mainline) and HTTP relays.

## Client Abstractions

The public client API separates configuration, I/O, caching, and lookup
freshness:

```text
ClientBuilder
├── cache: InMemoryCache | custom Cache | disabled
└── backend: DHT | HTTP relays | combined DHT + relays
        ↓
      Client
      ├── publish(&SignedPacket)
      └── resolve(&PublicKey, ResolvePolicy)
```

- `ClientBuilder` configures the cache, TTL bounds, timeouts, and available
  network backends. The default native client uses both the DHT and the default
  relays.
- `Client` is the cloneable async facade that coordinates publishing,
  resolution, cache updates, and consistent public errors.
- `Cache` is replaceable. Clients use an in-memory LRU by default; a configured
  cache size or custom-cache capacity of zero disables caching, and the
  `lmdb-cache` feature provides an opt-in persistent implementation.
- `ResolvePolicy` makes the source and freshness trade-off explicit:
  `CacheOnly` avoids DHT queries, `CacheFirst` returns the fastest fresh result
  without going backward from an expired cached packet, and `NetworkOnly`
  aggregates the configured networks for their most recent observed state.

The DHT, relay, and combined backend implementations are internal details.
Select them through `ClientBuilder` rather than depending on their concrete
types. With both backends enabled, publishing uses both concurrently;
`NetworkOnly` resolution waits for both and selects the most recent result,
while `CacheFirst` may finish as soon as a fresh result above the cache floor is
available.

See the [integration guide](https://github.com/Pubky/pkarr/blob/main/docs/integration.md)
for configuration examples and the
[API documentation](https://docs.rs/pkarr/latest/pkarr/) for the full public
surface.

## Runtime and Platform Support

- The client API is asynchronous.
- On native targets, `async_compat` allows use with non-Tokio executors.
- Browsers use the `relays` feature and Fetch-backed HTTP requests because they
  cannot access the UDP DHT directly.
- WASI is not supported.

## More Documentation

- [Quickstart](https://github.com/Pubky/pkarr/blob/main/docs/quickstart.md)
- [Feature reference](https://github.com/Pubky/pkarr/blob/main/docs/features.md)
- [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples)
