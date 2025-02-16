# Pkarr

Rust implementation of [Pkarr](https://pkarr.org) for publishing and resolving DNS packets over [Mainline DHT](https://github.com/Pubky/mainline).

## Documentation

- **[API Documentation](https://docs.rs/pkarr/latest/pkarr/)**
- **[Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples)**

## Features

### Runtime Support

- **Asynchronous by Default**: Built on async/await for optimal performance
- **Blocking API Available**: Use `Client::as_blocking()` for synchronous operations
- **Runtime Agnostic**: Compatible with non-Tokio runtimes via `async_compat`

### WebAssembly

- **Browser Environment**: Designed for JavaScript/Wasm integration
- **Relay Communication**: Uses browser's Fetch API for relay calls
- **Limitations**: 
  - Not compatible with WASI
  - Cannot use WASI bindings for direct DHT access
