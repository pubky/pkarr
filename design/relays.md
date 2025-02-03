# Relays

In UDP-less environments (like web browsers, virtual machines, containers and firewalled networks) clients need to rely on a remote server that can relay their [PUT](#PUT) and [GET](#GET) messages. 

## API

Public relays need to setup cors headers.

### PUT 

#### Request

```
PUT /:z-base32-encoded-key HTTP/2
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
If-Unmodified-Since: Fri, 18 Oct 2024 13:24:21 GMT

<body>
```

Body is described at [Payload](#Payload) encoding section.

On receiving a PUT request, the relay server should:
1. Encode the `seq` and `v` to a *bencode* message as follows: `3:seqi<sequence>e1:v<v's length>:<v's bytes>`
2. Verify that the `sig` matches the encoded message from step 1, if it is invalid, return a `400 Bad Request` response.
3. Perform the DHT Put request as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html), optionally using the `If-Unmodified-Since` as the CAS field.
4. If the DHT request is successful, return a `200 OK` response, otherwise if any error occured return a `500 Internal Server Error` response.

#### Errors

- `400 Bad Request` if the public key in the path is invalid, or the payload has invalid signature, or DNS packet.
- `409 Conflict` if the timestamp is older than what the server or the DHT network already seen (equivalent to error code `302` in `BEP0044`).
- `412 Precondition Failed` if the `If-Unmodified-Since` condition fails (equivalent to error code `301` in `BEP0044`).
- `413 Payload Too Large` if the payload is larger than 1072 bytes.
- `428 Precondition Required` if the server is already publishing another packet for the same key, it should require a `If-Unmodified-Since` header.
- `429 Too Many Requests` if the server is rate limiting requests from the same IP.

### GET

```
GET /:z-base32-encoded-key HTTP/2
If-Modified-Since: Fri, 18 Oct 2024 13:24:21 GMT
```

#### Response

```
HTTP/2 200 OK
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
Content-Type: application/pkarr.org/relays#payload
Cache-Control: public, max-age=300
Last-Modified: Fri, 18 Oct 2024 13:24:21 GMT

<body>
```

`Cache-Control` header would help browsers reduce their reliance on the relay, the `max-age` should be set to be the minimum `ttl` in the resource records in the packet or some minimum ttl chosen by the relay.
`If-Modified-Since` can be sent by the client to avoid downloading packets they already have, when the relay responds with `304 Not Modified`.

Body is described at [Payload](#Payload) encoding section.

On receiving a GET request, the relay server should:
1. Perform a DHT mutable GET query as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html)
2. Concat the `sig`, big-endian encoded `seq`, and `v`.
3. If no records were found, respond with `404 Not Found`. 

#### Errors

- `400 Bad Request` if the public key in the path is invalid.
- `404 Not Found` if the packet is not found.

## Payload

Relay payload is a subset of the [Canonical encoding](./base.md#Encoding), omitting the leading public key:

```abnf
RelayPayload = signature timestamp dns-packet

signature   = 64 OCTET ; ed25519 signature over encoded DNS packet
timestamp   =  8 OCTET ; big-endian UNIX timestamp in microseconds
dns-packet  =  * OCTET ; compressed encoded DNS answer packet, less than 1000 bytes
```

## Relation to resolvers

Organizations that elect to run a Relay, may choose to run their DHT node as [Resolver](./resolvers.md) as well.
