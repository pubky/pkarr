# Pkarr

Rust implementation of [Pkarr](pkarr.org).

Publish and resolve DNS packets over Mainline DHT.

**[API Docs](https://docs.rs/pkarr/latest/pkarr/)**

## Get started

Check the [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples).

## Using the Pkarr Client

#### Blocking API Support

By default, Pkarr client is designed to be asynchronous. However, if you prefer, you can easily obtain a blocking version of all methods by calling `Client::as_blocking()`. 

#### Compatibility with Non-Tokio Runtimes

This client utilizes Tokio, but it remains compatible with other futures-based libraries thanks to the `async_compat` crate. This ensures seamless integration with various asynchronous runtimes.

#### WebAssembly support

This version of Pkarr assumes that you are running Wasm in a JavaScript environment,
and calling relays over thew web browser Fetch API, so you can't use it in Wasi for example, 
nor can you use some Wasi bindings to use the DHT directly.
