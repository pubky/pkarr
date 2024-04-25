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

<sig><seq>[<v>]
```

Body should consist of:

- 64 bytes `sig`
- 8 bytes `u64` big-endian `seq` 
- 0-1000 bytes of `v`.

On receiving a PUT request, the relay server should:
1. Encode the `seq` and `v` to a *bencode* message as follows: `3:seqi<sequence>e1:v<v's length>:<v's bytes>`
2. Verify that the `sig` matches the encoded message from step 1, if it is invalid, return a `400 Bad Request` response.
3. Perform the DHT Put request as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html).
4. If the DHT request is successful, return a `200 OK` response, otherwise if any error occured return a `500 Internal Server Error` response.

### GET

```
GET /:z-base32-encoded-key HTTP/2
```

#### Response

```
HTTP/2 200 OK
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, PUT, OPTIONS
Cache-Control: public, max-age=300

<sig><seq>[<v>]
```

`Cache-Control` header would help browsers reduce their reliance on the relay, the `max-age` should be set to be the minimum `ttl` in the resource records in the packet or some minimum ttl chosen by the relay.

Body should consist of:

- 64 bytes `sig`
- 8 bytes `u64` big-endian `seq` 
- 0-1000 bytes of `v`.

On receiving a GET request, the relay server should:
1. Perform a DHT mutable GET query as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html)
2. Concat the `sig`, big-endian encoded `seq`, and `v`.
3. If no records were found, respond with `404 Not Found`. 


## Oportunities 

Relays are only expected to transparently relay messages to an from the DHT, but popular public relays may add caching and other useful features like indexing and search over all the records they have seen, effectively acting as a public phone book without having to crawl the web for Pkarr keys.
