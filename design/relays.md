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

<signature><timestamp>[<dns packet>]
```

Body is described at [Payload](#Payload) encoding section.

On receiving a PUT request, the relay server should:
1. Encode the `seq` and `v` to a *bencode* message as follows: `3:seqi<sequence>e1:v<v's length>:<v's bytes>`
2. Verify that the `sig` matches the encoded message from step 1, if it is invalid, return a `400 Bad Request` response.
3. Perform the DHT Put request as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html).
4. If the DHT request is successful, return a `200 OK` response, otherwise if any error occured return a `500 Internal Server Error` response.


#### Errors

- `400 Bad Request` if the public key in the path is invalid, or the payload has invalid signature, or DNS packet.
- `409 Conflict` if the timestamp is older than what the server or the DHT network already seen.
- `413 Payload Too Large` if the payload is larger than 1072 bytes
- `429 Too Many Requests` if the server is already publishing a packet for the same key, or if it is rate limiting requests from the same IP.

### GET

```
GET /:z-base32-encoded-key HTTP/2
```

#### Response

```
HTTP/2 200 OK
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
Content-Type: application/pkarr.org/relays#payload
Cache-Control: public, max-age=300

<signature><timestamp>[<dns packet>]
```

`Cache-Control` header would help browsers reduce their reliance on the relay, the `max-age` should be set to be the minimum `ttl` in the resource records in the packet or some minimum ttl chosen by the relay.

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

| part       |  length  |         Note                         |
| ---------- | -------- | ------------------------------------ | 
| signature  | 64       | ed25519                              |
| timestamp  | 8        | big-endian timestamp in microseconds |
| DNS packet | variable | compressed DNS answer packet.        |


## Relation to resolvers

Organizations that elect to run a Relay, may choose to run their DHT node as [Resolver](./resolvers.md) as well.
