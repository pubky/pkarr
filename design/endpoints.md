# Endpoints

As Pkarr is a new system, we recommend prioritizing the use of [SVCB and HTTPS (RFC 9460)](https://www.rfc-editor.org/rfc/rfc9460.pdf) records as the primary method for defining and discovering endpoints. For detailed terminology and specifications, please refer to the aforementioned RFC.

This approach enables servers to be accessible via URLs such as `https://<pkarr key>`.

## Client-Side Resolution

Clients can resolve the `SignedPacket` for a `<pkarr key>` to obtain the server's IP address through `A` or `AAAA` records. Additionally, the `HTTPS` record can provide the server's port number and other useful parameters, such as Encrypted Client Hello (ECH) and Application-Layer Protocol Negotiation (ALPN), which support HTTP/2 and HTTP/3.

Similarly, resolving a URL like `pubky://<pkarr key>` involves fetching the `SignedPacket` for `<pkarr key>` and searching for the `HTTPS` resource record named `_pubky.pxnu33x7jtpx9ar1ytsi4yxbp6a5o36gwhffs8zoxmbuptici1jy`.

### Resolution Algorithm

To clarify the process of resolving an endpoint, here is a step-by-step algorithm. While it largely follows standard DNS semantics, it simplifies certain aspects (e.g., ignoring CNAMEs) to aid implementation:

1. **Locate the `SignedPacket`** for the `qname` (the hostname or authority part of the URL). The TLD in `qname` must be a valid Pkarr public key.
2. **Identify all `HTTPS` records** within the resolved `SignedPacket` that match the `qname` or a wildcard pattern.
3. **Sort `HTTPS` records** in ascending order by their `priority` field to facilitate failover.
4. **Shuffle records** within each priority level to randomize lookup results and support load balancing.
5. **Examine the `target` field** of the first `HTTPS` record:
   - If the `target` is `.` (dot), the `SignedPacket` itself is the endpoint. Use the `A` and `AAAA` records for IP addresses, and optionally use `HTTPS` record parameters for connection establishment.
   - If the `target` is another Pkarr key, query that key for a `SignedPacket` and repeat from step (2). If resolution fails, proceed to the next `HTTPS` record in the list.
   - If the `target` is not a valid Pkarr key, assume it is an ICANN domain name. Use standard DNS resolvers to find the endpoint's IP address or delegate to a conventional HTTP client for resolution.

## Server-Side Configuration

### Publishing Records

Servers that are directly reachable should publish a `SignedPacket` with their keypair and create an `HTTPS` record at the apex (their public key) with the target `.`. This record should specify the server's `port` number and any other parameters that assist in establishing a direct connection.

These servers should also publish `A` and `AAAA` records to allow clients to resolve their IP addresses.

For servers running behind a reverse proxy or hosted by third parties, an `HTTPS` record can be created with the `target` field set to the domain name of the web host they wish to redirect to.

### Compatibility with Legacy Browsers

In environments or browsers that lack support for Pkarr or restrict network access to HTTP endpoints via specific APIs, only `targets` that are domain names can be used. This limitation arises because `A`/`AAAA` records alone cannot facilitate secure connections in such contexts.

To ensure compatibility, servers accessible via Pkarr endpoints must provide at least one endpoint associated with an ICANN domain. This ensures that legacy browsers can connect to the server by selecting this domain-based endpoint.
