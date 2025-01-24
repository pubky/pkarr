# Changelog

All notable changes to pkarr will be documented in this file.

## [Unreleased]

### Added

- Add `SignedPacket::builder()` and convenient methods to create `A`, `AAAA`, `CNAME`, `TXT`, `SVCB`, and `HTTPS` records.
- Add `SignedPacket::all_resource_records()` to access all resource records without accessing the dns packet.
- Use `pubky_timestamp::Timestamp` 
- Impl `PartialEq, Eq` for `SignedPacket`.
- Impl `From<PublicKey>` for `CacheKey`.
- Add `SignedPacket::serialize` and `SignedPacket::deserialize`.
- Add `SignedPacket::elapsed()` to return the number of seconds since this packet was last seen from remote network. 
- Derive `serde::Serialize` and `serde::Deserialize` for `SignedPacket`.
- Add `pkarr::LmdbCache` for persistent cache using lmdb.
- Add `pkarr.pubky.org` as an extra default Relay and Resolver.
- Add `Client::resolve_iter` and `Client::resolve_stream` to iterate or stream incoming `SignedPacket`s
- Add feature `endpoints` to resolve `HTTPS` and `SVCB` endpoints over Pkarr
- Add feature `reqwest-resolve` to create a custom `reqwest::dns::Resolve` implementation from `Client` and `relay::client::Client`
- Add feature `tls` to create `rustls::ClientConfig` from `Client` and `relay::client::Client` and create `rustls::ServerCongif` from `KeyPair`.
- Add feature `reqwest-builder` to create a `reqwest::ClientBuilder` from `Client` and `relay::client::Client` using custom dns resolver and preconfigured rustls client config.
- Implement `FromStr` for `PublicKey`
- Implement `TryFrom<MutableItem>` for `SignedPacket`
- Add `client::native::ClientBuilder::no_default_network` to disable both the `Dht` and `relays` settings, and allow you to choose what to add back.
- Add `client::native::ClientBuilder::use_mainline` to use default bootstrap nodes and resolvers and use the Mainline Dht. 
- Add `client::native::ClientBuilder::extra_resolvers` to use extra resolvers nodes to the default resolvers.
- Add `client::native::ClientBuilder::extra_bootstrap` to add extra nodes to the default bootstrap nodes.
- Add client::native::`ClientBuilder::use_relays` to cuse default relays and use relays to publish and resolve packets.
- Add client::native::`ClientBuilder::relays` to set (override) the relays to be used.
- Add client::native::`ClientBuilder::extend_relays` to add extra relay servers to the list of relays used.

### Changed

- Replace `Settings` with `Config` with public fields.
- Replace `z32` with *(base32)*.
- `SignedPacket::last_seen` is a `Timestamp` instead of u64.
- Make `rand` non-optional, and remove the feature flag.
- Replace `ureq` with `reqwest` to work with HTTP/2 relays, and Wasm.
- Update `mainline` to v5.0.0.
- `Client::shutdown` and `Client::shutdown_sync` are now idempotent and return `()`.
- `Client::resolve`, `Client::resolve_sync` and `relay::Client::resolve` return expired cached `SignedPacket` _before_ making query to the network (Relays/Resolvers/Dht).
- Update `simple-dns` so you can't use `Name::new("@")`, instead you should use `Name::new(".")`, `SignedPacket::resource_records("@")` still works.
- Replace `ClientBuilder::testnet()` with `ClientBuilder::bootstrap()`.
- `Client::cache()` returns an option, in case the cache size is set to zero.
- Rename feature `relay` to `relays`.
- Default client uses both `mainline` and `relays`, and each can be disabled with feature flags or the builder methods.
- Improve `Debug` and `Display` implementations for `SignedPacket`.

### Removed

- Remvoed `relay_client_web`, replaced with *(pkarr::Client)*.
- Removed `SignedPacket::from_packet`.
- Removed `SignedPacket::packet` getter.
- Removed rexported `mainline`
