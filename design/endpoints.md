# Endpoints

Given that Pkarr is a new system, we can prioritize the use of [SVCB and HTTPS (RFC 9460)](https://www.rfc-editor.org/rfc/rfc9460.pdf) records as the primary method for defining and discovering endpoints. Refer to the aforementioned RFC for detailed terminology.

This approach allows HTTPs servers to be reachable with URLs like `https://<pkarr key>`. Clients can resolve the `SignedPacket` for the `<pkarr key>` to obtain the server's IP address through `A` or `AAAA` records. Additionally, the `HTTPS` record can provide the server's port number and other beneficial parameters like encrypted client hello and ALPN (supporting http2/http3).

Similarly, resolving a URL like `pubky://<pkarr key>` involves fetching the `SignedPacket` for `<pkarr key>` and searching for the `HTTPS` resource record named `_pubky.pxnu33x7jtpx9ar1ytsi4yxbp6a5o36gwhffs8zoxmbuptici1jy`.

## Resolution Algorithm

To clarify the process of resolving an endpoint, here's a step-by-step algorithm. It mostly adheres to standard DNS semantics, but it makes some simplifications (like ignoring CNAMEs), thus it's provided here to aid implementations:

1. **Locate the `SignedPacket`** for the `qname` (the hostname or authority part of the URL). The TLD in `qname` has to be a valid pkarr public key.
2. **Identify all `HTTPS` records** within the resolved `SignedPacket` with that `qname`, or matches the `qname` if it is a wildcard.
3. **Sort `HTTPS` records** ascendingly by their `priority` field, to facilitate failover.
4. **Shuffle records** within each priority level to randomize lookup results and support load balancing.
5. **Examine the `target` field** of the first `HTTPS` record:
   - If the `target` is `.` (dot), the `SignedPacket` itself is the endpoint. Use the `A` and `AAAA` records for IP addresses, and optionally use `HTTPS` record parameters for connection establishment.
   - If the `target` is another Pkarr key, query that key for a `SignedPacket` and repeat from step (2). If resolution fails, proceed to the next `HTTPS` record in the list.
   - If the `target` is not a valid Pkarr key, assume it's an ICANN domain name. Use standard DNS resolvers to find the endpoint's IP address or delegate to a conventional HTTP client for resolution.

## Compatibility with Legacy Browsers

In environments or browsers that lack support for Pkarr or restrict network access to HTTP endpoints via specific APIs, only `targets` that are domain names can be utilized. This limitation exists because `A`/`AAAA` records alone cannot facilitate secure connections in such contexts.

Therefore, servers accessible via Pkarr endpoints must ensure at least one endpoint is associated with an ICANN domain. This ensures that legacy browsers can connect to the server by selecting this domain-based endpoint.
