use std::env;
use std::process;

fn main() {
    #[cfg(any(
        feature = "endpoints",
        feature = "reqwest-resolve",
        feature = "tls",
        feature = "reqwest-builder"
    ))]
    {
        if env::var("TARGET")
            .ok()
            .map(|t| t.starts_with("wasm32"))
            .unwrap_or_default()
        {
            if !cfg!(feature = "relays") {
                eprintln!("Pkarr Build Error: `relays` feature must be enabled for WASM builds.");
                process::exit(1);
            }

            if cfg!(feature = "dht") {
                eprintln!("Pkarr Build Error: `dht` feature must be disabled for WASM builds.");
                process::exit(1);
            }
        } else if !cfg!(any(feature = "dht", feature = "relays")) {
            eprintln!(
                "Pkarr Build Error: At least one of `dht` or `relays` features must be enabled."
            );
            process::exit(1);
        }
    }
}
