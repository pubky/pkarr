# Endpoints

As Pkarr is a new system, we recommend prioritizing [`SVCB` and `HTTPS` (RFC 9460)](https://www.rfc-editor.org/rfc/rfc9460.pdf) records for defining and discovering endpoints. This enables URLs like: `https://<pkarr key>`

## Client-Side Resolution

### Overview
Clients resolve the `SignedPacket` for a `<pkarr key>` to obtain:
- Server IP address via `A`/`AAAA` records 
- Port number and parameters via `HTTPS` record, including:
  - Encrypted Client Hello (`ECH`)
  - Application-Layer Protocol Negotiation (`ALPN`) for `HTTP/2` and `HTTP/3`

For URLs like `pubky://<pkarr key>`, clients fetch the `SignedPacket` and search for the `HTTPS` record named `_pubky.pxnu33x7jtpx9ar1ytsi4yxbp6a5o36gwhffs8zoxmbuptici1jy`

### Resolution Algorithm

To clarify the process of resolving an endpoint, here is a step-by-step algorithm. While it largely follows standard DNS semantics, it simplifies certain aspects (e.g., ignoring `CNAME`s) to aid implementation:

1. **Locate `SignedPacket`**
   - Fetch packet for `qname` (hostname/URL authority)
   - TLD must be valid Pkarr public key

2. **Process `HTTPS` Records**
   - Identify all matching `HTTPS` records for `qname` or wildcard pattern
   - Sort by priority (ascending) for failover
   - Shuffle within priority levels for load balancing

3. **Handle Target Field**
   - If target is "." (dot):
     - Use `SignedPacket` as endpoint
     - Use `A`/`AAAA` records for IP addresses
     - Use `HTTPS` parameters for connection setup
   - If target is Pkarr key:
     - Query that key's `SignedPacket`
     - Repeat from step 2
     - On failure, try next `HTTPS` record
   - If target is not Pkarr key:
     - Treat as ICANN domain
     - Use standard DNS resolution

## Server-Side Configuration

### Direct Access Servers
Servers that are directly reachable should:
1. Publish `SignedPacket` with their keypair
2. Create `HTTPS` record at apex (public key) with:
   - Target set to "."
   - Port number specified
   - Any other connection parameters included
3. Publish `A` and `AAAA` records for IP resolution

### Proxy/Hosted Servers
For servers behind reverse proxy or third-party hosting:
- Create `HTTPS` record
- Set target to web host domain name

### Legacy Browser Compatibility
1. **Environment Limitations**
   - Only domain name targets work in some contexts
   - `A`/`AAAA` records alone cannot secure connections
   - Some browsers restrict network access to `HTTP` endpoints

2. **Requirements**
   - Servers must provide at least one ICANN domain endpoint
   - This ensures legacy browsers can connect via domain-based endpoint
