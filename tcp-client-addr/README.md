# TCP client address

This crate answers one question for each accepted TCP connection: which client
address should the application use? It does not serve HTTP or apply rate limits.

- `IdentityMode::Direct` returns the TCP peer as the client. HTTP forwarding
  headers have no effect.
- `IdentityMode::ProxyProtocol` requires a PROXY v1/v2 preface from a configured
  trusted immediate peer. It rejects missing, malformed, unsupported, and
  oversized address headers. A complete preface must arrive within one second by
  default. It never silently falls back to the TCP peer.

Proxy mode deliberately rejects v1 `UNKNOWN` and v2 `LOCAL`. Those prefaces
do not supply a forwarded TCP client address, so accepting them would make a
per-client-IP decision ambiguous. Configure proxy health checks accordingly.

PROXY v1 prefaces are limited to 107 bytes by the protocol. The configurable
header-size limit also applies to v1, and governs the maximum v2 preface size.
Any v2 bytes after the address block are consumed but ignored. TLV structure,
contents, and optional CRC32C checksums are not validated; TLVs never influence
the returned client address.

Both modes return the **original `TcpStream`**, positioned at the first
application byte, plus both the client and immediate peer addresses. The reader
uses exact reads and never buffers application bytes, so no replay stream or
HTTP-server integration is necessary.
`ClientAddr::client()` preserves the received address. Use
`ClientAddr::client_ip()` as a per-IP rate-limit key, or
`ClientAddr::normalized_client()` when an HTTP server expects a socket address:
both collapse IPv4-mapped IPv6 to IPv4 without changing native IPv6 addresses.

The identity belongs to the entire TCP connection. A proxy must not reuse or
multiplex one backend connection for different clients: every request or HTTP/2
stream on it would receive the same identity. This includes HTTP proxies with
shared upstream connection pools, even when the proxy itself is trusted.
See the [PROXY protocol specification](https://www.haproxy.org/download/2.9/doc/proxy-protocol.txt).

For TLS listeners, process the connection in this order: accept TCP, call
`identify`, perform the TLS handshake on the returned stream, then serve the
application protocol. The PROXY preface precedes TLS and must be consumed first.

```rust,no_run
use tcp_client_addr::{IdentityMode, ProxyProtocol};
use ipnet::IpNet;
use tokio::net::TcpListener;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let listener = TcpListener::bind("127.0.0.1:8080").await?;
let trusted_proxy: IpNet = "127.0.0.1/32".parse()?;
let mode = IdentityMode::ProxyProtocol(ProxyProtocol::new([trusted_proxy])?);

let (stream, _) = listener.accept().await?;
let (stream, addr) = mode.identify(stream).await?;
println!("client: {}, immediate peer: {}", addr.client(), addr.peer());
println!("rate-limit IP: {}", addr.client_ip());
// Give `stream` to any TCP-based HTTP or other protocol server.
drop(stream);
# Ok(())
# }
```

The built-in PROXY-preface timer starts when `identify` begins reading. Use
`ProxyProtocol::with_header_timeout` to change its one-second default. The
caller should still bound concurrent identification tasks; an idle peer
otherwise holds one. An application may also apply an acceptance-based
deadline for its entire connection setup and first HTTP request. A trusted CIDR
is an authorization boundary: any peer admitted by it can claim any source
address. Restrict access to the listener with a firewall or private network;
do not expose it to direct clients or trust a shared network containing
untrusted processes. A loopback CIDR still trusts every local process that can
connect to the listener. For a proxy chain, the immediate proxy must send the
PROXY preface; trust is configured for that immediate peer, not for every
upstream hop. The first proxy must establish the real client address rather
than passing through a client-supplied forwarding header.

Run the regular tests with `cargo test -p tcp-client-addr`. An
optional end-to-end test uses a real Nginx `stream` proxy to verify the PROXY
source address and untouched application bytes:

```bash
cargo test -p tcp-client-addr --test nginx_proxy -- --ignored
```

That test requires Docker, a local `nginx:alpine` image, and Docker host
networking. It starts and removes its own container.
