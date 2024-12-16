# Changelog

All notable changes to pkarr will be documented in this file.

## [Unreleased]

### Added

- Add feature `endpoints` to resolve `HTTPS` and `SVCB` endpoints over Pkarr
- Add feature `reqwest-resolve` to create a custom `reqwest::dns::Resolve` implementation from `Client` and `relay::client::Client`
- Add feature `tls` to create `rustls::ClientConfig` from `Client` and `relay::client::Client` and create `rustls::ServerCongif` from `KeyPair`.
- Add feature `reqwest-builder` to create a `reqwest::ClientBuilder` from `Client` and `relay::client::Client` using custom dns resolver and preconfigured rustls client config.

### Changed

- Replace `Settings` with `Config` with public fields.

##  [3.0.0](https://github.com/pubky/mainline/compare/v2.2.0...v3.0.0) - 2024-12-05

### Added

- Add `SignedPacket::builder()` and convenient methods to create `A`, `AAAA`, `CNAME`, `TXT`, `SVCB`, and `HTTPS` records.
- Add `SignedPacket::all_resource_records()` to access all resource records without accessing the dns packet.
- Use `pubky_timestamp::Timestamp` 
- Impl `PartialEq, Eq` for `SignedPacket`.
- Impl `From<PublicKey>` for `CacheKey`.
- Add `SignedPacket::serialize` and `SignedPacket::deserialize`.
- Derive `serde::Serialize` and `serde::Deserialize` for `SignedPacket`.
- Add `pkarr::LmdbCache` for persistent cache using lmdb.
- Add `pkarr.pubky.org` as an extra default Relay and Resolver.
- Make `Client::resolve_rx` public to be able to stream or iterate over incoming `SignedPacket`s

### Changed

- replace `z32` with *(base32)*.
- `SignedPacket::last_seen` is a `Timestamp` instead of u64.
- make `rand` non-optional, and remove the feature flag.
- replace `ureq` with `reqwest` to work with HTTP/2 relays, and Wasm.
- update `mainline` to v3.0.0
- `Client::shutdown` and `Client::shutdown_sync` are now idempotent and return `()`.
- `Client::resolve`, `Client::resolve_sync` and `relay::Client::resolve` return expired cached `SignedPacket` _before_ making query to the network (Relays/Resolvers/Dht).
- Export `Settings` as client builder.
- Update `simple-dns` so you can't use `Name::new("@")`, instead you should use `Name::new(".")`, `SignedPacket::resource_records("@")` still works.

### Removed

- Remvoed `relay_client_web`, replaced with *(pkarr::relay::Client)*.
- Removed `SignedPacket::from_packet`.
- Removed `SignedPacket::packet` getter.
