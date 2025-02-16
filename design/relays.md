# Relays

In UDP-less environments (like web browsers, virtual machines, containers and firewalled networks), clients need to rely on a remote server that can relay their [PUT](#PUT) and [GET](#GET) messages. 

## API Overview

Public relays need to set up CORS headers.

## Endpoints

### PUT Operation

Stores a DNS packet in the DHT network through the relay.

#### Request

```http
PUT /:z-base32-encoded-key HTTP/2
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
If-Unmodified-Since: Fri, 18 Oct 2024 13:24:21 GMT

<body>
```

#### Response

```http
HTTP/2 204 NO CONTENT
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS	
```

> The `body` is described in the [Payload](#Payload) encoding section.

#### Processing Steps

On receiving a PUT request, the relay server should:

1. Encode the `seq` and `v` to a *bencode* message:
   ```
   3:seqi<sequence>e1:v<v's length>:<v's bytes>
   ```

2. Verify that the `sig` matches the encoded message from step 1. If it is invalid, return a `400 Bad Request` response.

3. Perform the DHT Put request as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html), optionally using the `If-Unmodified-Since` as the CAS field.

4. If the DHT request is successful, return a `204 No Content` response. Otherwise, if any error occurred, return a `500 Internal Server Error` response.

#### Error Responses

| Status Code | Description |
|------------|-------------|
| `400 Bad Request` | Invalid public key, signature, or DNS packet |
| `409 Conflict` | Timestamp is older than existing data ([BEP0044](https://www.bittorrent.org/beps/bep_0044.html) code `302`) |
| `412 Precondition Failed` | `If-Unmodified-Since` condition fails ([BEP0044](https://www.bittorrent.org/beps/bep_0044.html) code `301`) |
| `413 Payload Too Large` | Payload exceeds 1072 bytes |
| `428 Precondition Required` | Server requires `If-Unmodified-Since` header |
| `429 Too Many Requests` | Rate limit exceeded for IP |

---

### GET Operation

Retrieves a DNS packet from the DHT network through the relay.

#### Request

```http
GET /:z-base32-encoded-key HTTP/2
If-Modified-Since: Fri, 18 Oct 2024 13:24:21 GMT
```

#### Response

```http
HTTP/2 200 OK
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
Content-Type: application/pkarr.org/relays#payload
Cache-Control: public, max-age=300

Last-Modified: Fri, 18 Oct 2024 13:24:21 GMT

<body>
```

> The `body` is described in the [Payload](#Payload) encoding section.

#### Caching

- The `Cache-Control` header helps browsers reduce relay dependency
- `max-age` should be set to the minimum `ttl` in the resource records or relay-defined minimum TTL
- `If-Modified-Since` can be sent by the client to avoid downloading packets they already have, when the relay responds with `304 Not Modified`.

#### Processing Steps

On receiving a GET request, the relay server should:

1. Perform a DHT mutable GET query as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html)
2. Concatenate the `sig`, big-endian encoded `seq`, and `v`
3. If no records were found, respond with `404 Not Found`

#### Error Responses

| Status Code | Description |
|------------|-------------|
| `400 Bad Request` | Invalid public key in path |
| `404 Not Found` | Packet not found |

---

## Payload Format

Relay payload is a subset of the [Canonical encoding](./base.md#Encoding), omitting the leading public key:

```abnf
RelayPayload = signature timestamp dns-packet

signature   = 64 OCTET ; ed25519 signature over encoded DNS packet
timestamp   =  8 OCTET ; big-endian UNIX timestamp in microseconds
dns-packet  =  * OCTET ; compressed encoded DNS answer packet, less than 1000 bytes
```
