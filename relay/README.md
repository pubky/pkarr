# Pkarr Relay

A server that functions as a [pkarr relay](https://pkarr.org/relays).

## Installation & Usage

### Local Installation

Build the binary:
```bash
cargo build --release
```

### Running the Relay

#### Basic Usage
```bash
../target/release/pkarr-relay
```

#### With Custom Logging
```bash
../target/release/pkarr-relay -t=pkarr=debug,tower_http=debug
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
      - .pkarr_cache:/cache
    command: pkarr-relay
```

2. Using Docker directly:
   - Mount the cache directory as a volume
   - The `.pkarr_cache` directory will store cached keys permanently
