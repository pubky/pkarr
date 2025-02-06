use cfg_aliases::cfg_aliases;

fn main() {
    // Convenience aliases
    cfg_aliases! {
        wasm_browser: { all(target_family = "wasm", target_os = "unknown") },
        dht: { all(feature = "dht", not(target_family = "wasm")) },
        relays: { feature = "relays" },
        client: {
            any(
                all(
                    not(target_family = "wasm"),
                    any(feature = "dht", feature = "relays")
                ),
                all(
                    target_family = "wasm",
                    feature = "relays"
                )
            )
        },
    }
}
