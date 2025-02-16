# TLS

Once an endpoint (server) for a Pkarr domain has been resolved (see [Endpoints](https://pkarr.org/endpoints)), and assuming the endpoint itself is listening on its own `Pkarr key`, we can use that key to establish a TLS connection using [Raw Public Key (RFC 7250)](https://www.rfc-editor.org/rfc/rfc7250).

## Supported Algorithm

Since Pkarr keys are `Ed25519` keys, the signature scheme `0x0807` (`Ed25519`) should be the only supported algorithm for TLS connections.

## Client-Side Implementation

If you have full control over your client, you can resolve the endpoint first and use the endpoint's `public key` to establish a TLS connection using the Raw Public Key method.

If your client relies on middleware that validates server certificates, you can resolve endpoints as usual and verify whether any endpoint matches the server's `hostname`.

## Server-Side Configuration

### Reverse Proxies

When using a reverse proxy with Pkarr servers, proper configuration is essential to maintain end-to-end encryption and client information. Here's how to set it up:

1. **Architecture Overview**:
   ```
   [Client] <---> [Reverse Proxy :8443] <---> [Pkarr Server :3000]
                     |
                     +-- Dedicated port (8443)
                     +-- TLS passthrough (no termination)
                     +-- Proxy Protocol or proxy_pass
   ```

2. **Key Components**:
   - The reverse proxy listens on a dedicated port (e.g. `8443`)
   - TLS connections pass through without termination
   - Client IP and protocol information are preserved
   - Traffic is forwarded to the Pkarr server's internal port

3. **HTTPS Record Example**:
   ```json
   {
     "type": "HTTPS",
     "target": "proxy.example.com",  // Reverse proxy address
     "port": 8443,                   // Dedicated proxy port
     "alpn": ["h2", "http/1.1"],     // Supported protocols
     "priority": 1,                  // Lower values have higher priority
   }
   ```

4. **Sample HAProxy Configuration**:
   [HAProxy](https://www.haproxy.org/) is a popular open-source reverse proxy software. Here's how to configure it for Pkarr:


   ```
   frontend pkarr_frontend
     bind *:8443
     mode tcp
     option tcplog
     default_backend pkarr_backend

   backend pkarr_backend
     mode tcp
     server pkarr_server 127.0.0.1:3000 send-proxy-v2
   ```

This configuration ensures that:
- TLS connections remain end-to-end encrypted
- Client information is preserved through the proxy
- The Pkarr server receives unmodified TLS handshakes
- The `HTTPS` record correctly points clients to the proxy

### Compatibility with Legacy Browsers

In environments or browsers that lack support for Pkarr or restrict network access to `HTTP` endpoints via specific APIs, supporting TLS with Raw Public Keys will be impossible. To ensure compatibility, servers must also provide an endpoint accessible through a traditional `ICANN` domain with a valid TLS certificate. This ensures that legacy browsers can establish secure connections.
