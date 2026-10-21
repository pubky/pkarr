# pkresolve

Pubky endpoint resolution from signed PKARR packets. This workspace crate
depends on `pkarr` for packet types and parsing. Packet providers own network
I/O; the resolver has no HTTP stack or async runtime dependency.

Imported from `pubky/pkresolve` at commit
`96e6ec73d3bc021e1f4a799f3a5b4db4bc9b30d2`. The MIT license and copyright
are preserved in [LICENSE](LICENSE).

## Bundle client

Enable the optional `service-client` feature to use `pkresolve::ServiceClient`
for packet caching and bundle processing. It is disabled by default:

```toml
pkresolve = { path = "path/to/pkarr/pkresolve", features = ["service-client"] }
```

Call `prepare(request)` to obtain cached endpoints or a `BundleRequest`.
Send its JSON body to `/v1/resolve` using your HTTP stack, then pass the retained
request and response bytes to `accept_bundle()`. Failed bundles leave the cache
unchanged; packets retain their native `last_seen`.

See the [prototype limits](#status).

## Packet providers

Derive Pubky endpoints from a `PacketProvider`, without an HTTP stack or async
runtime:

```rust
let resolution = pkresolve::derive_resolution(&request, &provider)?;
```

Providers supply valid `Arc<SignedPacket>` values and own validation, caching,
and proof collection. Each key must return the same packet version throughout
one resolution. A `HashMap<PublicKey, Arc<SignedPacket>>` implements the trait.

For asynchronous packet retrieval, implement `AsyncPacketProvider` and await
the asynchronous entry point:

```rust
let resolution =
    pkresolve::derive_resolution_async(&request, &async_provider).await?;
```

Both entry points share routing and freshness checks. The synchronous entry
point needs no executor. The async entry point awaits each packet lookup;
dropping its future drops the pending lookup. Providers own I/O timeouts.

Supports same-name HTTPS ServiceMode, multiple endpoints, signed addresses and
hints, HTTP/1.1, and HTTP/2. Aliases, CNAME routing, alternate targets, TLS
delegation, CA TLS, ECH, and HTTP/3 are not implemented. TLS delegation records
are ignored.

Use `resolution.service` as TLS SNI and require its raw public key for every
candidate. Preserve the original HTTP authority. Check `resolution.expires_at`
before reuse; it follows the packet's shortest TTL, aged from `last_seen`.

Run the direct ServiceMode example from the repository root:

```sh
cargo run -p pkresolve --example resolve_pubky --offline
```

## Status

Prototype. The bundle client calls `SignedPacket::deserialize()`, which parses
stored packets without verifying their signatures. Do not accept untrusted
bundles until signature verification is implemented.

Known client cache limitations:

- Expired cached packets can still produce terminal errors, such as
  `NoCompatibleEndpoint` for an absent scoped service. Those errors do not
  trigger a fresh bundle fetch and can persist until the packet is replaced.
- Incoming packet versions are not compared with previously accepted versions.
  A delayed response can return older routes and overwrite newer cached packets.

## Development

Run from the workspace root:

```sh
cargo test -p pkresolve
cargo test -p pkresolve --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

The [Chromium bridge](../bindings/chromium/README.md) keeps its CXX toolchain in
a separate Cargo workspace. The empty upstream server stub is not included;
HTTP serving belongs in `pkarr-relay`.

The crate keeps its own version. Releases publish `pkarr`, then `pkresolve`,
then `pkarr-relay`. Configure crates.io trusted publishing for `pkresolve`
before using the release workflow. In-workspace consumers should use a path
and version dependency, for example:

```toml
pkresolve = { path = "../pkresolve", version = "0.1" }
```
