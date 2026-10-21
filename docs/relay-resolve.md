# Resolve Pubky services through a relay

`POST /v1/resolve` returns a packet bundle for a Pubky service. It uses
the workspace's [pkresolve crate](../pkresolve/README.md) and matches its
`ServiceClient` wire format. Both serialize `pkresolve::ResolutionRequest`
directly and use `pkresolve::service_protocol::ResolveBundle` for responses.

## Request

Send JSON with `Content-Type: application/json`. Replace `<key>` below with
the service owner's canonical z-base-32 public key:

```json
{
  "service": "app.<key>",
  "url_port": 443,
  "supported_alpn": ["h2", "http/1.1"]
}
```

- `service` is a DNS service name, such as `<key>`, `app.<key>`, or
  `_pubky.<key>`. These are distinct names. URLs and the `pubky` display prefix
  are not accepted. ASCII case and one trailing root dot are normalized.
- `url_port` is the original URL port, from 1 through 65535. For ports other
  than 443, PKResolve looks up `_<port>._https.<service>`.
- `supported_alpn` must contain at least one of `h2` and `http/1.1`, in client
  preference order. Other protocols are rejected; duplicates are removed.

All fields are required. Unknown fields are rejected. The request body limit
is 1,104 bytes, shared with ordinary relay requests.

## Response

A successful response has status `200`, content type `application/json`, and
`Cache-Control: no-store`:

```json
{
  "schema": 1,
  "packets": ["<base64-encoded native signed packet>"],
  "resolution": {
    "service": "app.<key>",
    "url_port": 443,
    "candidates": [
      {
        "addresses": ["203.0.113.10"],
        "port": 443,
        "priority": 1,
        "protocols": ["h2", "http/1.1"]
      }
    ],
    "valid_for_ms": 270000
  }
}
```

The bundle contains every owner packet requested by the resolver, once per
key. Each string is standard base64 of `SignedPacket::serialize()`, including
its original `last_seen` timestamp. Packet order has no meaning. This is not
the raw payload returned by `GET /{key}`. Serving a cached packet does not
advance its observation time.

`resolution` is the relay's computed result, included for convenience. It
contains the original service and URL port, candidates in priority order, and
the remaining lifetime when the summary was created, rounded down to
milliseconds. `valid_for_ms` does not account for transport delay and must not
be restarted on receipt. The process-local `Resolution::expires_at` timestamp
is not sent.

The `resolution` field is required and included for convenience. Clients should
verify it by deriving their own resolution for the original request, using
the bundled packets as the packet source. `ServiceClient` derives routes and
expiry this way rather than trusting the relay's summary. Bundles with a
missing or null `resolution` are rejected.

Use `ServiceClient::prepare` to construct the request and retain the returned
`BundleRequest`. Pass it and the response bytes to
`ServiceClient::accept_bundle` to derive endpoint candidates. The
client's native packet decoder does not verify signatures; consumers must
verify signatures separately before trusting responses from an untrusted
relay. A successful resolution also does not authenticate the remote TLS peer.

## Supported behavior

The endpoint delegates routing decisions to PKResolve's implementation:

- Same-name HTTPS ServiceMode records, with signed addresses or address hints.
- HTTP/1.1 and HTTP/2, with direct raw-key TLS authentication using the
  destination service's public key.
- Exact service scope and port handling; addresses alone do not enable a route.

HTTPS AliasMode follows exact target names within Pubky namespaces. Every
packet used is included in the bundle, and the earliest packet expiry limits
the result. The original service identity is retained. Loops and chains longer
than 32 names are rejected. Multiple aliases select the first in signed packet
order so clients can reproduce the resolution. Alias parameters are ignored.

Routing CNAMEs, public DNS alias targets, alternate ServiceMode targets,
CA authentication, ECH, and
HTTP/3 are unsupported. TLS delegation records are currently ignored;
they do not change the required destination TLS key. The endpoint returns
packets, not an HTTP redirect or proxied application content.

Alternate-target HTTPS ServiceMode records are skipped when selecting routes.
They do not prevent usable same-service records from resolving. If all
protocol-compatible ServiceMode records use alternate targets, resolution
returns `422` with `unsupported resolution feature: alternate HTTPS target`.

Lookups use the relay's existing LMDB cache and `CacheFirst` policy, including
its configured minimum and maximum TTLs. Cache misses and entries expired
under that policy trigger a DHT lookup. PKResolve independently checks signed
record TTLs: if the relay's minimum TTL keeps an entry cached beyond its signed
TTL, resolution returns `502` without forcing a refresh.
Each traversal retains the first packet version it reads for each key, so
shared-cache updates cannot change that traversal's proof. One async mutex
serializes packet lookups within a traversal, including cache and DHT access.
Repeated lookups reuse the retained result. Other requests proceed independently
and can observe newer versions in the shared cache.

The existing HTTP quota applies to every request. The user DHT quota applies
only when a lookup contacts the DHT. Resolution has a 30-second deadline after
body parsing, in addition to the existing 30-second request body deadline.
Each traversal may request up to 32 distinct owner keys, bounding lookup work
and bundle size. Failed lookups count toward this limit. Repeated lookups of
the same key return the same result without consuming another slot. The relay
can collect multiple packets; the PKResolve implementation currently requests
only one and still has the routing limitations described above.

## Errors

Failures return a non-success status and, when available, a plain-text reason:

- `400`: malformed JSON, invalid field types, missing or unknown fields,
  invalid service name or port, empty protocol list, or unsupported ALPN.
- `404`: no packet found. Invalid DHT packets also return the existing
  `pkarr-invalid-signed-packet-seq` diagnostic header when applicable.
- `413`: request body exceeds the limit.
- `415`: request does not have a JSON content type.
- `422`: resolution finds unsupported routing, a disabled service, no compatible
  endpoint, no usable addresses, or exceeds the 32-owner lookup limit.
- `429`: HTTP or user DHT quota exhausted.
- `502`: malformed or expired resolution evidence.
- `503`: DHT lookup unavailable.
- `504`: resolution deadline exceeded.
- `500`: an unexpected internal resolution failure.

No error proves nonexistence or authorizes weaker TLS authentication.
