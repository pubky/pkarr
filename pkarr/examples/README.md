# Examples

## Publish 

```sh
cargo run --example publish
```

## Resolve Eager

```sh
cargo run --example resolve <zbase32 public key from Publish step>
```

## Resolve Most Recent

```sh
cargo run --example resolve-most-recent <zbase32 public key from Publish step>
```

## Relay

```sh
cargo run --example relay-client
```

## Async Publish 

```sh
cargo run --example async-publish --features="async"
```

## Async Resolve Eager

```sh
cargo run --example async-resolve --features="async" <zbase32 public key from Publish step>
```

## Resolve Most Recent

```sh
cargo run --example async-resolve-most-recent --features="async" <zbase32 public key from Publish step>
```

## Relay

```sh
cargo run --example async-relay-client --features="relay async"
```
