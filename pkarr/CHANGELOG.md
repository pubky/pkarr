# Changelog

All notable changes to pkarr will be documented in this file.

## [Unreleased]

### Added

- Use `pubky_timestamp::Timestamp` 
- Impl `PartialEq, Eq` for `SignedPacket`.
- Impl `From<PublicKey>` for `CacheKey`.
- Derive `serde::Serialize` and `serde::Deserialize` for `SignedPacket`.
- Add `SignedPacket::serialize` and `SignedPacket::deserialize`.
- Add `SignedPacket::elapsed()` to return the number of seconds since this packet was last seen from remote network. 
- Add `SignedPacket::all_resource_records()` to access all resource records without accessing the dns packet.
- Add `SignedPacket::builder()` and convenient methods to create `A`, `AAAA`, `CNAME`, `TXT`, `SVCB`, and `HTTPS` records.
- Add `SignedPacket` implemention of `TryFrom<MutableItem>`.
- Add `pkarr::LmdbCache` for persistent cache using lmdb.
- Add `pkarr.pubky.org` as an extra default Relay and Resolver.
- Implement `FromStr` for `PublicKey`
- Add `ClientBuilder::no_default_network()` to disable both the `Dht` and `relays` settings, and allow you to choose what to add back.
- Add `ClientBuilder::no_dht()` to disable using the DHT, relaying only on relays. 
- Add `ClientBuilder::dht()` to access `mainline::DhtBuilder` and control the internal dht node.
- Add `ClientBuilder::extra_resolvers()` to use extra resolvers nodes to the default resolvers.
- Add `ClientBuilder::extra_bootstrap()` to add extra nodes to the default bootstrap nodes.
- Add `ClientBuilder::no_relays()` to disable using relays and relay only on the DHT. 
- Add `ClientBuilder::relays()` to set (override) the relays to be used.
- Add `ClientBuilder::extend_relays()` to add extra relay servers to the list of relays used.
- Add `Client::dht()` to return a reference to the internal dht node if configured. 
- Add `Client::resolve_most_reecent()` to do exactly what it says on the tin. 
- Add `PublishError` enum with more granular errors for concurrent publishing.
- Add feature `endpoints` to resolve `HTTPS` and `SVCB` endpoints over Pkarr
- Add feature `reqwest-resolve` to create a custom `reqwest::dns::Resolve` implementation from `Client` and `relay::client::Client`
- Add feature `tls` to create `rustls::ClientConfig` from `Client` and `relay::client::Client` and create `rustls::ServerCongif` from `KeyPair`.
- Add feature `reqwest-builder` to create a `reqwest::ClientBuilder` from `Client` and `relay::client::Client` using custom dns resolver and preconfigured rustls client config.

### Changed

- Replace `Settings` with non-consuming `ClientBuilder`.
- `SignedPacket::last_seen` is a `Timestamp` instead of u64.
- Replace `z32` with `base32`.
- Update `mainline` to v5.0.0.
- Make `rand` non-optional, and remove the feature flag.
- Replace `ureq` with `reqwest` to work with HTTP/2 relays, and Wasm, but you can still use the client outside tokio.
- Update `simple-dns` so you can't use `Name::new("@")`, instead you should use `Name::new(".")`, `SignedPacket::resource_records("@")` still works.
- `Client::resolve`, `Client::resolve_sync` return expired cached `SignedPacket` _before_ making query to the network (Relays/Resolvers/Dht) in the background.
- Replace `ClientBuilder::testnet()` with `ClientBuilder::bootstrap()`.
- `Client::cache()` returns an option, in case the cache size is set to zero.
- Rename feature `relay` to `relays`.
- Default client uses both `mainline` and `relays`, and each can be disabled with feature flags or the builder methods.
- Improve `Debug` and `Display` implementations for `SignedPacket`.
- update to `mainline` v5.
- `Client::publish()` takes an optional `cas` argument.

### Removed

- Remvoed `relay_client_web`, replaced with *(pkarr::Client)*.
- Removed `SignedPacket::from_packet`.
- Removed `SignedPacket::packet` getter.
- Removed rexported `mainline`
- Removed `Client::shutdown` and `Client::shutdown_sync`, and replaced with shutdown on `Drop`.
- Removed crate level `Error` enum, and replaced with more granular error types.
