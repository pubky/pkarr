# Pkarr Relay

A server that functions as a [pkarr](https://pkarr.org) [relay](https://pkarr.org/relays).

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
../target/release/pkarr-relay --config=./config.toml
```

You can customize logging levels

```bash
../target/release/pkarr-relay --config=./config.toml -t=pkarr=debug,tower_http=debug
```

## Using Docker
To build and run the Pkarr relay using Docker, this repository has a `Dockerfile` in the top level. You could use a small `docker-compose.yml` such as:

```
services:
  pkarr:
    container_name: pkarr
    build: .
    volumes: 
      - ./config.toml:/config.toml
      - .pkarr_cache:/cache
    command: pkarr-relay --config=/config.toml
```
Alternatively, lunch docker correctly attaching the `config.toml` as a volume in the right location. In the example above `.pkarr_cache` relative directory is used to permanently store pkarr cached keys.

An example `./config.toml` can be copied from `./src/config.example.toml` and customized as needed.

This will make the Pkarr relay accessible at http://localhost:6881.
