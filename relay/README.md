# Pkarr Relay

A server that functions as a [pkarr relay](https://pkarr.org/relays).

## Installation & Usage

### Installation

```bash
cargo install pkarr-relay
```

### Running the Relay

#### With Custom Configuration
```bash
pkarr-relay --config=./config.toml
```

You can find an example configuration file [here](https://github.com/pubky/pkarr/blob/main/relay/src/config.example.toml).

#### With Custom Logging
```bash
pkarr-relay  -t=pkarr=debug,tower_http=debug
```

Once running, the Pkarr relay will be accessible at http://localhost:6881 (unless the config file specifies another port number).

### Docker Deployment

To build and run the Pkarr relay using Docker:

```yaml
# docker-compose.yml
services:
  pkarr:
    container_name: pkarr
    build: .
    volumes:
      - ./config.toml:/config.toml
      - .pkarr_cache:/cache
    command: pkarr-relay --config=/config.toml
```

Alternatively, launch Docker directly with the `config.toml` attached as a volume. In the example above, `.pkarr_cache` is used to permanently store cached keys.

Copy the example configuration from `./src/config.example.toml` and customize as needed.
