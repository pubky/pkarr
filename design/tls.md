# TLS

Once an endpoint (server) for a Pkarr domain has been resolved (see [Endpoints](https://pkarr.org/endpoints)), and assuming the endpoint itself is listening on its own Pkarr key, we can use that key to establish a TLS connection using [Raw Public Key (RFC 7250)](https://www.rfc-editor.org/rfc/rfc7250).

## Supported Algorithm

Since Pkarr keys are ed25519 keys, the signature scheme `0x0807` (Ed25519) should be the only supported algorithm for TLS connections.

## Client-Side Implementation

If you have full control over your client, you can resolve the endpoint first and use the endpoint's public key to establish a TLS connection using the Raw Public Key method.

If your client relies on middleware that validates server certificates, you can resolve endpoints as usual and verify if any endpoint matches the server's hostname.

## Server-Side Configuration

### Reverse Proxies

If the server is running behind a reverse proxy (which likely does not support Pkarr) and cannot be addressed directly, the following steps should be taken:

1. **Dedicated Port**: The reverse proxy should listen on a dedicated `port` and forward traffic to the server's IP and port.
2. **TLS Passthrough**: The reverse proxy should **not** terminate TLS encryption. Instead, it should use a `proxy_pass` configuration or the [PROXY Protocol](https://www.haproxy.com/blog/use-the-proxy-protocol-to-preserve-a-clients-ip-address) if the server supports it.
3. **HTTPS Record Mapping**: Map the server's `HTTPS` record's `target` to the reverse proxy's address.
4. **Port Mapping**: Map the server's `HTTPS` record's `port` to the reverse proxy's dedicated port.

### Compatibility with Legacy Browsers

In environments or browsers that lack support for Pkarr or restrict network access to HTTP endpoints via specific APIs, supporting TLS with Raw Public Keys will be impossible. To ensure compatibility, servers must also provide an endpoint accessible via a traditional ICANN domain with a valid TLS certificate. This ensures that legacy browsers can establish secure connections.
