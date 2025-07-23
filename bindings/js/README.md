# JS Pkarr bindings

Wasm-pack wrap of [pkarr](https://github.com/pubky/pkarr) client.

## How To Build/Test the NPM Package

1. Create a binary to run a testnet mainline DHT and Pkarr relay. Go to the root folder and `cargo build`
2. Go to `bindings/js/pkg`
3. Run `npm run build`
5. Run tests with `npm run test`
6. EXTRA: If you want to run example, `npm run example`