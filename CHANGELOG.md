# Changelog

All notable changes to pkarr client and server will be documented in this file.

## [Unreleased]

### Added

- Add strict monotonic unix `Timestamp`.
- Impl `PartialEq, Eq` for `SignedPacket`.
- Impl `From<PublicKey>` for `CacheKey`.

### Changed

- replace `z32` with *(base32)*.
- `SignedPacket::last_seen` is a `Timestamp` instead of u64.
- make `rand` non-optional, and remove the feature flag.
- replace `ureq` with `reqwest` to work with HTTP/2 relays, and Wasm.
- update `mainline` to v3.0.0
- `Client::shutdown` and `Client::shutdown_sync` are now idempotent and return `()`.
- `Client::resolve`, `Client::resolve_sync` and `relay::Client::resolve` return expired cached `SignedPacket` if nothing else was found from the network.

### Removed

- Remvoed `relay_client_web`, replaced with *(pkarr::relay::Client)*.
