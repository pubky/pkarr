# Examples

## Publish 

```sh
cargo run --example publish
```

or to use a Relay client:

```sh
cargo run --features relay --example publish
```

## Resolve

```sh
cargo run --example resolve <zbase32 public key from Publish step>
```

or to use a Relay client:

```sh
cargo run --features relay --example resolve <zbase32 public key from Publish step>
```

## HTTP

Run an HTTP server listening on a Pkarr key

```sh
cargo run --features endpoints --example http-serve
```

An HTTPS url will be printend with the Pkarr key as the TLD, paste in another terminal window:

```sh
cargo run --features reqwest-resolve --example http-get <url>
```

And you should see a `Hello, World!` response.
