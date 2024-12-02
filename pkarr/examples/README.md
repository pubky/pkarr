# Examples

## Publish 

```sh
cargo run --example publish
```

## Resolve

```sh
cargo run --example resolve <zbase32 public key from Publish step>
```

## HTTP

Run an HTTP server listening on a Pkarr key

```sh
cargo run --features endpoints --example http-serve <ip address> <port number>
```

An HTTPs url will be printend with the Pkarr key as the TLD, paste in another terminal window:

```sh
cargo run --features reqwest-resolve --example http-get <url>
```

And you should see a `Hello, World!` response.
