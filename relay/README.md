# Pkarr Relay

A server that functions as a [pkarr relay](https://github.com/pubky/pkarr/blob/main/design/relays.md).

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
