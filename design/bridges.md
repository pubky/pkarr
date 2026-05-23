# Bridges

Optional spec for declaring back-pointers from a Pkarr identity to a human-readable
name held in another naming system.

Pkarr deliberately does not solve human-readable naming. As stated in the
[introduction](../docs/introduction.md), petname systems, personal phonebooks,
and traditional DNS records pointing at a Pkarr key are all valid ways to layer
human-friendly handles on top of a sovereign keypair.

This spec defines a single, low-cost convention so that those external naming
systems can be discovered _from_ a Pkarr packet, without Pkarr taking on any
dependency on them.

## Goals

- Allow a Pkarr publisher to advertise that they also control a name in an
  external naming system (e.g. a traditional DNS zone, a blockchain-based name,
  a fediverse handle).
- Allow applications that already speak that external system to perform
  bidirectional verification: the external record points at the Pkarr key, and
  the Pkarr packet points back at the external name. Either half on its own
  proves nothing; together they prove that whoever controls the keypair also
  controls the external name at publish time.
- Keep Pkarr clients agnostic. No client is required to resolve any external
  system. Bridges are opt-in at the application layer.

## Non-Goals

- Pkarr does not endorse, host, or resolve any external naming system.
- Pkarr does not define how external systems should encode their own
  back-pointer to a Pkarr key. That belongs in each external system's own
  conventions.
- Pkarr does not promote any specific external system over another.

## Record Format

Publishers MAY include one or more `TXT` records at the name `_bridge` in their
`SignedPacket`. Each record advertises one external name in one external naming
system.

The record value is a space-separated set of `key=value` pairs. Two keys are
required:

| Key      | Required | Meaning                                                                                    |
|----------|----------|--------------------------------------------------------------------------------------------|
| `system` | yes      | Short identifier for the external naming system. Lowercase ASCII. See registry below.       |
| `name`   | yes      | The publisher's name in that system, in whatever canonical form the system itself defines. |

Additional `key=value` pairs MAY be present and SHOULD be ignored by
implementations that do not recognise them.

### Example

A publisher who also controls the traditional DNS name `alice.example`:

```
_bridge  TXT  "system=dns name=alice.example"
```

A publisher who also controls the Namecoin `.bit` name `alice`:

```
_bridge  TXT  "system=namecoin name=d/alice"
```

A publisher who advertises both:

```
_bridge  TXT  "system=dns name=alice.example"
_bridge  TXT  "system=namecoin name=d/alice"
```

## System Identifiers

The `system` value identifies which external naming system the `name` lives in.
This document defines two initial values; further systems can be added by
amending this spec or by application-level convention.

| `system`   | External system                              | Canonical `name` form                         |
|------------|----------------------------------------------|-----------------------------------------------|
| `dns`      | Traditional ICANN-rooted DNS                 | A fully-qualified domain name without trailing dot. |
| `namecoin` | Namecoin name database                       | The on-chain key including its namespace prefix, e.g. `d/alice` for the domain namespace, `id/alice` for identity. |

Implementations that consume `_bridge` records MUST treat unknown `system`
values as opaque and ignore them rather than failing.

## Verification

A `_bridge` record on its own asserts nothing. It is an unverified claim by the
keypair. Verification requires that the external naming system itself contains
a matching back-pointer to the Pkarr public key.

The exact form of that back-pointer is defined by the external system, not by
this spec. For example:

- For `system=dns`, an application might require a `TXT` record at
  `_pkarr.alice.example` whose value is the z-base-32 Pkarr public key.
- For `system=namecoin`, an application might require that the JSON value
  stored at `d/alice` on the Namecoin chain contains a `pkarr` field whose
  value is the z-base-32 Pkarr public key.

When both halves are present and agree, an application can treat the binding
as proven for as long as both records remain current. Republishing cadence
and freshness requirements are the application's responsibility.

## Security Considerations

- A `_bridge` record is only as trustworthy as the external system it points
  at. Applications MUST NOT treat an unverified bridge claim as proof of
  anything beyond "the keypair asserted this name."
- External naming systems may have very different liveness, finality, and
  revocation models. Applications consuming `_bridge` records SHOULD make
  their trust assumptions about each `system` value explicit.
- Pkarr packets are size-limited to 1000 bytes (see [base](./base.md)).
  Publishers should not flood their packet with bridge claims.

## Privacy Considerations

Bridge records are public. Publishers who do not want their Pkarr key linked
to a particular external name should not publish a `_bridge` record for it.
