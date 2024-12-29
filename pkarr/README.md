# Pkarr

Rust implementation of [Pkarr](pkarr.org).

Publish and resolve DNS packets over Mainline DHT.

**[API Docs](https://docs.rs/pkarr/latest/pkarr/)**

## Get started

Check the [Examples](https://github.com/Nuhvi/pkarr/tree/main/pkarr/examples).

## WebAssembly support

This version of Pkarr assumes that you are running Wasm in a JavaScript environment,
and using the Relays clients, so you can't use it in Wasi for example, nor can you
use some Wasi bindings to use the DHT directly. If you really need Wasi support, please
open an issue on `https://git.pkarr.org`.
