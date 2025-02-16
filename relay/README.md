# Pkarr Relay

A server that functions as a [pkarr relay](https://pkarr.org/relays).

## Installation & Usage

### Local Installation

1. Build the binary:
```bash
cargo build --release
```

2. Copy and customize the config file (optional):
```bash
cp src/config.example.toml config.toml
```

### Running the Relay

#### Basic Usage
```bash
../target/release/pkarr-relay --config=./config.toml
```

#### With Custom Logging
```bash
../target/release/pkarr-relay --config=./config.toml -t=pkarr=debug,tower_http=debug
```

Once running, the Pkarr relay will be accessible at http://localhost:6881.

### Using Docker

Alternatively, you can run the relay using Docker. This repository includes a `Dockerfile` and you can run it in two ways:

1. Using docker-compose (recommended):
```yaml
services:
  pkarr:
    container_name: pkarr
    build: .
    volumes: 
      - ./config.toml:/config.toml
      - .pkarr_cache:/cache
    command: pkarr-relay --config=/config.toml
```

2. Using Docker directly:
   - Copy the example config: `cp src/config.example.toml config.toml`
   - Mount the config file and cache directory as volumes
   - The `.pkarr_cache` directory will store cached keys permanently
