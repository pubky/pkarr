# PKResolve for Chromium

Rust owns caching and bundle processing behind synchronous CXX calls.
Chromium owns network requests and their asynchronous lifetime. Full Chromium
GN integration and the standalone pinned CXX build remain unverified.

## Resolution flow

Create a Rust client with `new_client()` and keep it on one Chromium sequence.

1. Call `prepare(service, url_port, protocols)`.
2. If `needs_bundle()` is false, consume the preparation with
   `prepared_result()` to obtain cached endpoints or a terminal error.
3. Otherwise, copy `request_body()` and POST it to the configured HTTPS
   `/v1/resolve` endpoint using Chromium's network stack. Retain the preparation
   until the response arrives.
4. On HTTP success, pass the same preparation and complete response bytes to
   `accept_bundle()`. Rust verifies packet signatures and derives endpoints
   before updating the cache.

The request JSON contains `service`, `url_port`, and `supported_alpn`. The
schema-1 response requires `packets` and a `resolution` summary. Packets are
standard base64 of native `SignedPacket::serialize()` bytes, including their
original `last_seen`. Bundles can contain multiple packets in any order, with
one packet per owner. Incomplete bundles fail without another HTTP fetch.
Signature verification covers the publication timestamp and DNS contents;
`last_seen` is preserved unsigned metadata, not proof of freshness.

The summary contains `service`, `url_port`, `tls_public_key`, `candidates`, and
`valid_until_unix_ms` (absolute Unix milliseconds). It is included for
convenience: the client derives its own result from the bundled packets and
the retained request. Failed bundle acceptance leaves the cache unchanged.

Chromium handles HTTPS certificate checks, HTTP errors, cancellation, and
caller lifetime. An HTTP failure discards the preparation; no Rust operation
is waiting to be cancelled. Rust exposes no HTTP interface or polling API.

Chromium also owns destination connections. Use the resolved service as SNI,
preserve the original HTTP authority, and require possession of the destination
raw Ed25519 key. HTTPS AliasMode delegates this key to the final routing
owner; the service name remains the original HTTP authority and SNI.
The bridge exports the delegated key as `Resolution.public_key`, a 32-byte
array. `url_port` remains the original URL port; each candidate's `port` is
the connection port and can differ.

Sample the monotonic clock before calling `prepare()` for a cached result, or
`accept_bundle()` for a fetched result. Add the returned `valid_for_ms` to that
sample to obtain a reuse deadline. Copying results or crossing IPC must not
extend it. This duration comes from the locally derived monotonic deadline,
not the summary's `valid_until_unix_ms`. It uses the earliest packet deadline
in the alias chain, retaining the age recorded in each packet's `last_seen`.
The bundle validation limits in the
[resolver status](../../pkresolve/README.md#status) still apply before accepting
untrusted bundles.

## Build inside Chromium

[Chromium recommends CXX for Rust/C++ integration][chromium-ffi]. Place this
repository at `//third_party/pkarr`. [BUILD.gn](BUILD.gn) builds `:pkarr`,
`:pkresolve`, and `:bridge` from this checkout. Do not vendor a second published
PKARR crate or keep a replacement `//third_party/rust/pkarr/v8:lib` target.
Update any direct users of that old label to
`//third_party/pkarr/bindings/chromium:pkarr`.

### Prepare external Rust dependencies

Merge this fragment into Chromium's
`third_party/rust/chromium_crates_io/Cargo.toml`. Extend existing entries rather
than defining them twice; preserve features required by other Chromium users.
These version requirements match the packet and resolver Cargo manifests.
Chromium's Cargo lockfile pins the resolved external dependency versions.

```toml
[dependencies]
base32 = "0.5"
base64 = "0.22"
bytes = "1"
document-features = "0.2"
ed25519-dalek = { version = "3", default-features = false }
getrandom = { version = "0.4", default-features = false }
httpdate = "1"
self_cell = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
simple-dns = "0.12"
thiserror = "2"
```

Merge the following entries into
`third_party/rust/chromium_crates_io/gnrt_config.toml`. For existing tables,
add missing fields and append to lists; preserve existing safety classifications,
shipping restrictions, visibility entries, and `allow_unsafe` settings. The
`sandbox` and `shipped = false` entries below retain the prototype's restrictions;
they are not approval to ship these dependencies in Chromium.

```toml
[crate.base32]
group = "sandbox"
shipped = false
allow_first_party_usage = false
extra_input_roots = ["../README.md"]
[crate.base32.extra_kv]
allow_unsafe = false
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

[crate.document-features]
group = "sandbox"
shipped = false
allow_first_party_usage = false
[crate.document-features.extra_kv]
allow_unsafe = false
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

[crate.ed25519-dalek]
group = "sandbox"
shipped = false
allow_first_party_usage = false
[crate.ed25519-dalek.extra_kv]
allow_unsafe = false
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

[crate.httpdate]
group = "sandbox"
shipped = false
allow_first_party_usage = false
[crate.httpdate.extra_kv]
allow_unsafe = false
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

[crate.self_cell]
group = "sandbox"
shipped = false
allow_first_party_usage = false
[crate.self_cell.extra_kv]
allow_unsafe = true
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

[crate.simple-dns]
group = "sandbox"
shipped = false
allow_first_party_usage = false
extra_input_roots = ["../README.md"]
[crate.simple-dns.extra_kv]
allow_unsafe = false
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]

# getrandom already has Chromium policy and input settings. Add this field
# to its existing extra_kv table without replacing the other fields.
[crate.getrandom.extra_kv]
visibility = ["//third_party/pkarr/bindings/chromium:pkarr"]
```

The narrow visibility additions let the packet target use crates otherwise
restricted to `//third_party/rust/*`. The remaining dependencies retain their
existing visibility. Additional `README.md` inputs cover dependency
`include_str!` calls; the upstream PKARR target separately declares its own
`pkarr/README.md` input.

Run Chromium's dependency tooling from the Chromium source directory:

```sh
python3 tools/crates/run_gnrt.py vendor
python3 tools/crates/run_gnrt.py gen
```

Use the toolchain required by your Chromium revision. Inspect the generated
GN diff: keep existing restrictions and ensure the restricted crates expose
only the additional `:pkarr` consumer. Do not hand-edit generated build files.

### Connect the bridge

Add `//third_party/pkarr/bindings/chromium:bridge` to the C++ caller's `deps`
and include the generated header:

```cpp
#include "third_party/pkarr/bindings/chromium/src/lib.rs.h"
```

Chromium supplies matching CXX runtime and generator versions through
`cxx_bindings`; do not substitute the standalone generator in a GN build.
The standalone Cargo workspace has its own CXX pin.

The packet target enables only `signed_packet`, leaving networking disabled.
GN does not run `pkarr/build.rs`: on native targets this feature selection
leaves the `wasm_browser`, `client`, `dht`, and `relays` aliases false. Revisit
the source list, dependencies, and aliases before changing that feature set.
The target uses third-party warning settings and omits Chromium's prelude,
while retaining its injected allocator dependency.

Cargo expands feature dependencies automatically. GN requires explicit
features: the resolver target enables both `service-client` and
`service-protocol` and includes `service_protocol.rs` in its source list.

## Run tests

Run the bridge's Rust tests from this repository:

```sh
cargo test --manifest-path bindings/chromium/Cargo.toml
```

The tests cover endpoint conversion, a two-packet AliasMode chain with a
delegated TLS key, distinct URL and connection ports, preserved packet age,
cache reuse, and malformed-bundle rejection without caching. The alias test
also supplies an incorrect summary to check that the bridge derives its
result from packets and the original request. They call the Rust operations
exposed by the bridge; they do not exercise a C++ caller or TLS connections.

After placing this checkout in Chromium and vendoring its dependencies, build
the same tests through GN to check Chromium's feature and dependency settings.
From the Chromium source directory, using your configured output directory:

```sh
autoninja -C out/Default pkresolve_chromium_unittests
out/Default/pkresolve_chromium_unittests
```

Cargo enables transitive features automatically; GN requires them to be listed
explicitly. Passing the Cargo tests alone does not validate the GN build.

## Build the standalone C++ example

See the [example source](examples/resolve.cc).

Run these Linux commands from the repository root with Rust, a C++17 compiler,
and crates.io access for the first build. The separate Cargo manifest pins
CXX 1.0.194 for Rust 1.85 compatibility.

```sh
cargo install cxxbridge-cmd --version 1.0.194 --locked
cargo build --manifest-path bindings/chromium/Cargo.toml --release
mkdir -p bindings/chromium/target/cxx
cxxbridge bindings/chromium/src/lib.rs --header \
  -o bindings/chromium/target/cxx/pkresolve.h
cxxbridge bindings/chromium/src/lib.rs \
  -o bindings/chromium/target/cxx/pkresolve.cc
c++ -std=c++17 -fno-exceptions -Ibindings/chromium/target/cxx \
  bindings/chromium/examples/resolve.cc \
  bindings/chromium/target/cxx/pkresolve.cc \
  bindings/chromium/target/release/libpkresolve_chromium.a \
  -ldl -pthread -lm -o bindings/chromium/target/cxx/resolve
packet_key=$(cargo run --quiet --manifest-path bindings/chromium/Cargo.toml \
  --example make_bundle -- /tmp/pkresolve-bundle.json)
bindings/chromium/target/cxx/resolve "$packet_key" /tmp/pkresolve-bundle.json
```

The example reads a response file, resolves `203.0.113.10:443`, then resolves
again from the internal cache. It prints one bundle read for two resolutions.
The packet's `last_seen` is ten seconds old; receipt preserves that age. The
example does not establish a TLS connection.

[chromium-ffi]: https://chromium.googlesource.com/chromium/src/+/refs/heads/main/docs/rust/ffi.md
