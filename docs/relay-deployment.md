# Relay client addresses and rate limits

The relay's `http.client_identity` setting determines which client IP is used
for HTTP request limits and user-initiated DHT operation limits. Incoming DHT
traffic continues to use the DHT peer's address.

The default mode, `direct`, uses the TCP peer address and ignores forwarding
headers. Behind a reverse proxy, all visitors share that proxy's IP-based
limits.

## Configure a trusted proxy

To preserve individual client IPs, configure your proxy to overwrite a header
with the client's IP address. In your relay configuration file, remove any
existing `client_identity` inline setting from `[http]`. Add this table after
the other `[http]` settings:

```toml
[http.client_identity]
mode = "http_header"
header = "X-Real-IP"
trusted_proxies = ["127.0.0.1/32"]
```

Use the header your proxy overwrites and the immediate proxy's TCP source
address or network. `trusted_proxies` must contain at least one IPv4 or IPv6
CIDR network. An allowed proxy can claim any client IP. Connections from
other peers are closed before HTTP parsing.

Every request must supply exactly one value for the configured header: a bare
IPv4 or IPv6 address without a port. Missing, duplicate, or malformed values
receive `400 Bad Request`, including on the index route, fallback routes,
and CORS preflights. This requirement also applies when rate limiting is
disabled. Address lists such as `X-Forwarded-For` chains are unsupported, and
there is no fallback to other headers or the TCP peer address.

Identity is resolved separately for every request, including requests that
share an HTTP/1 connection or HTTP/2 session. IPv4-mapped IPv6 addresses share
limits with their IPv4 equivalents.

For Cloudflare Tunnel, set `header = "CF-Connecting-IP"` and trust only the
local connector's TCP source address. Health checks must also connect from a
trusted address and supply the configured header in this mode.

Restart the relay after changing its configuration. See the
[example configuration](../relay/src/config.example.toml) for all settings.

## Migrate from `behind_proxy`

The `rate_limiter.behind_proxy` setting has been removed. Existing
configurations containing it fail to load rather than silently changing
which client IP is used.

- If it was `false`, remove it to keep using the TCP peer address.
- If it was `true`, remove it and configure `http.client_identity` as shown
  above. Ensure your proxy overwrites the selected header with one client IP.

For Rust callers, remove `behind_proxy` from `RateLimiterConfig` and use
`RelayBuilder::client_identity` with `ClientIdentityConfig::HttpHeader` to
configure header mode. Omitting the builder setting keeps direct mode.
