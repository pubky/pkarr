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

