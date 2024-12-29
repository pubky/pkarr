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
- Derive `serde::Serialize` and `serde::Deserialize` for `SignedPacket`.
- Add `pkarr::LmdbCache` for persistent cache using lmdb.
- Add `pkarr.pubky.org` as an extra default Relay and Resolver.
- Make `Client::resolve_rx` public to be able to stream or iterate over incoming `SignedPacket`s
- Add feature `endpoints` to resolve `HTTPS` and `SVCB` endpoints over Pkarr
- Add feature `reqwest-resolve` to create a custom `reqwest::dns::Resolve` implementation from `Client` and `relay::client::Client`
- Add feature `tls` to create `rustls::ClientConfig` from `Client` and `relay::client::Client` and create `rustls::ServerCongif` from `KeyPair`.
- Add feature `reqwest-builder` to create a `reqwest::ClientBuilder` from `Client` and `relay::client::Client` using custom dns resolver and preconfigured rustls client config.
- Implement `FromStr` for `PublicKey`
- Implement `TryFrom<MutableItem>` for `SignedPacket`
- Add `resolvres_to_socket_addrs()` function.

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
- Change `SignedPacket::to_relay_payload()` to `SignedPacket::as_relay_payload()`.
- Replace `bytes::Bytes` return types with `&[u8]`.

### Removed

- Remvoed `relay_client_web`, replaced with *(pkarr::relay::Client)*.
- Removed `SignedPacket::from_packet`.
- Removed `SignedPacket::packet` getter.
- Removed rexported `mainline`
- Remove `bytes` dependency.
