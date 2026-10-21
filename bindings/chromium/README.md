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
   `accept_bundle()`. Rust derives endpoints before updating the cache.

Chromium handles HTTPS certificate checks, HTTP errors, cancellation, and
caller lifetime. An HTTP failure discards the preparation; no Rust operation
is waiting to be cancelled. Rust exposes no HTTP interface or polling API.

Chromium also owns destination connections. Use the resolved service as SNI,
preserve the original HTTP authority, and require possession of the destination
raw Ed25519 key.

Sample the monotonic clock before calling `prepare()` for a cached result, or
`accept_bundle()` for a fetched result. Add the returned `valid_for_ms` to that
sample to obtain a reuse deadline. Copying results or crossing IPC must not
extend it. The bundle validation limits in the
[resolver status](../../pkresolve/README.md#status) still apply before accepting
untrusted bundles.

## Build inside Chromium

[Chromium recommends CXX for Rust/C++ integration][chromium-ffi].

1. Place this repository at `//third_party/pkarr`.
2. Vendor these dependencies with Chromium's `gnrt` workflow and generate their
   GN targets:

   ```toml
   [dependencies]
   thiserror = "2"
   base64 = "0.22"
   serde = { version = "1", features = ["derive"] }
   serde_json = "1"

   [dependencies.pkarr]
   version = "8.1"
   default-features = false
   features = ["signed_packet"]
   ```

3. Add `//third_party/pkarr/bindings/chromium:bridge` to the C++ caller's `deps`.
4. Include the generated header:

   ```cpp
   #include "third_party/pkarr/bindings/chromium/src/lib.rs.h"
   ```

[BUILD.gn](BUILD.gn) defines library and bridge targets.
Chromium supplies matching CXX runtime and generator versions through
`cxx_bindings`.

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
