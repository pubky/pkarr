# Pkarr Relay

The `pkarr-relay` binary is an HTTP gateway and cache for publishing and
resolving Pkarr packets through the Mainline DHT. See
[how Pkarr relays work](../design/relays.md).

## Installation

```bash
cargo install pkarr-relay
```

From a source checkout, inspect all CLI options with:

```bash
cargo run -p pkarr-relay -- --help
```

## Running the Relay

Run with defaults:

```bash
pkarr-relay
```

By default, the HTTP server binds to `0.0.0.0:6881`, which exposes it on all
IPv4 interfaces. It is available locally at `http://localhost:6881`, but it may
also be reachable from the LAN or Internet. Apply appropriate firewall or
reverse-proxy access controls on network-reachable hosts. The configuration
file can change the port.

Copy and edit the bundled
[example configuration](./src/config.example.toml), then pass its path:

```bash
pkarr-relay --config ./config.toml
```

Without `cache.path`, the relay creates a new cache in a temporary directory on
each run. Configure a stable path if the cache should survive restarts. The
example file also documents DHT, TTL, and rate-limit settings.

Configure logging with a tracing filter:

```bash
pkarr-relay --tracing-env-filter pkarr_relay=debug,tower_http=debug
```
