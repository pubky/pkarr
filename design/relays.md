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

<signature><message>
```

On receiving a PUT request, the relay server should:
1. Decode the payload such that 
    - the first *64 bytes* represent the **signature**
    - the remaining bytes represent the signed **message**
2. Decode the `seq` and `v` from the signed message, as they should be encoded using *bencode* as follows: `3:seqi<seq>e1:v<len>:<records encoded>`
3. Verify the signature, if it is invalid, return a `400 Bad Request` response.
4. Perform the DHT Put request as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html)
5. If the DHT request is successful, return a `200 OK` response, otherwise if any error occured return a `500 Internal Server Error` response.

### GET

```
GET /:z-base32-encoded-key HTTP/2
```

#### Response

```
HTTP/2 200 OK

<signature><message>
```

On receiving a GET request, the relay server should:
1. Perform a DHT mutable GET query as defined in [BEP0044](https://www.bittorrent.org/beps/bep_0044.html)
2. If the DHT request is successful, encode the `seq` and `v` as described in BEP0044 to represent the signed message, and respond with the concatenated `<signature><message>` body.
3. If no records were found, respond with `404 Not Found`. 


## Oportunities 

Relays are only expected to transparently relay messages to an from the DHT, but popular public relays may add caching and other useful features like indexing and search over all the records they have seen, effectively acting as a public phone book without having to crawl the web for Pkarr keys.
