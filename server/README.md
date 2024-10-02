# Pkarr Server

A server that functions as a [pkarr](https://github.com/Nuhvi/pkarr/) [relay](https://pkarr.org/relays) and
[resolver](https://pkarr.org/resolvers).

## Usage

Build

```bash
cargo build --release
```

Optinally Copy config file

```bash
cp src/config.example.toml config.toml
```

Run with an optional config file

```bash
../target/release/pkarr-server --config=./config.toml
```
