# Pkarr Server

A server that functions as a [pkarr](https://github.com/Nuhvi/pkarr/) [relay](https://pkarr.org/relays) and
[resolver](https://pkarr.org/resolvers).

## Usage

Build

```bash
cargo build --release
```

Run with an optional config file

```bash
../target/release/pkarr-server --config=./src/config.toml
```
