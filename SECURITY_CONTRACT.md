# Security contract

Pkarr maps Ed25519 identities to signed DNS records. The Rust library publishes
and resolves them; the relay bridges HTTP and DHT; JS/WASM bindings use relays.
Pubky uses this to discover homeservers. Keys control publication and may
authenticate services; failures can misdirect connections or deny discovery.

Owner: [@andrei-21](https://github.com/andrei-21). Reviewed: 2026-11-13.
Implementation baseline: `0322f88`. These are intended guarantees under the
conditions below, not a completed audit.

## From a key to a trusted connection

Peers, relays, and publishers may be hostile. Network resolution must verify
that records belong to the requested key. This establishes the signer's
authority over those records, not the safety of their contents or destinations.
Applications choose trusted identities and enforce authorization and
destination restrictions; Pkarr does not make those decisions.

Verification uses `ed25519-dalek`, secure key generation, `mainline` for DHT
authentication, and `simple-dns` parsing. Callers protect signing keys.
Once records enter a [cache](pkarr/src/client/cache.rs), the client trusts
their validity and key association. Cache implementations and storage therefore
form part of the security boundary, alongside process and OS integrity.

Discovery can delegate to another identity. Optional native
[TLS verification](pkarr/src/extra/tls.rs) authenticates a resolved endpoint's
key using `rustls`, not necessarily the original identity or a fixed snapshot
of discovery records. Applications must handle endpoint changes and failures.
Browser service connections instead rely on conventional HTTPS and its
certificate trust model. See [supported configurations](docs/integration.md)
and [TLS compatibility](design/tls.md).

## Authenticity does not establish freshness or availability

The signer chooses the timestamp, including past or future values; a signature
does not prove publication time. Cache freshness also depends on local time and
unsigned observation metadata that relays can misreport. Choosing a
[resolution policy](pkarr/src/types/resolve_policy.rs) is the caller's choice:
valid results can be stale, and concurrent lookups need not return in order.
Shared-cache writers must meet the
[coordination requirements](pkarr/src/client/builder.rs).

Records are public, with no confidentiality, global freshness, durable storage,
or availability guarantee. Publishers must republish; consumers must tolerate
withheld records and outages. Relay limits constrain hostile requests but cannot
prevent every denial of service. Operators own capacity, host and transport
protection, including the [proxy trust boundary](relay/src/rate_limiting.rs).

## Evidence and maintenance

The linked API docs define detailed obligations. Dependency correctness and
environment integrity are assumptions; [locked versions](Cargo.lock) and
[regression tests](pkarr/src/client/tests.rs) provide context and evidence only
for the versions and cases tested. No independent security audit is cited.

Report vulnerabilities via [GitHub](https://github.com/pubky/pkarr/security).
The owner updates this contract and API docs when guarantees, assumptions,
obligations, or supported configurations change.
