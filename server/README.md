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

You can customize logging levels

```bash
../target/release/pkarr-server --config=./config.toml -t=pkarr=debug,tower_http=debug
```

## Using Docker
To build and run the Pkarr server using Docker, this repository has a `Dockerfile` in the top level. You could use a small `docker-compose.yml` such as:

```
services:
  pkarr:
    container_name: pkarr
    build: .
    volumes: 
      - ./config.toml:/config.toml
      - .pkarr_cache:/cache
    command: pkarr-server --config=/config.toml
```
Alternatively, lunch docker correctly attaching the `config.toml` as a volume in the right location. In the example above `.pkarr_cache` relative directory is used to permanently store pkarr cached keys.

An example `./config.toml` here (we are mounting it on the container)
```
relay_port = 6881
dht_port = 6881
cache_path = "/cache"
cache_size = 1_000_000
resolvers = []
minimum_ttl = 300
maximum_ttl = 86400
[rate_limiter]
behind_proxy = false
per_second = 2
burst_size = 10
```

This will make the Pkarr server accessible at http://localhost:6881.