# Pkarr

Rust implementation of [Pkarr](pkarr.org).

Publish and resolve DNS packets over Mainline DHT.

**[API Docs](https://docs.rs/pkarr/latest/pkarr/)**

## Get started

Check the [Examples](https://github.com/Pubky/pkarr/tree/main/pkarr/examples).

## WebAssembly support

This version of Pkarr assumes that you are running Wasm in a JavaScript environment,
and calling relays over thew web browser Fetch API, so you can't use it in Wasi for example, 
nor can you use some Wasi bindings to use the DHT directly.
