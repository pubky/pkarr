use std::env;
use std::io;
use std::process::Command;

// If the process hangs, try `cargo clean` to remove all locks.

fn main() -> io::Result<()> {
    println!("🏗️ Building Pkarr WASM Package...");

    build_wasm("nodejs")?;
    patch()?;
    println!("📦 Pkarr WASM Package built successfully!");

    Ok(())
}

fn build_wasm(target: &str) -> io::Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(io::Error::other)?;

    let status = Command::new("wasm-pack")
        .env(
            "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
            "--cfg=getrandom_backend=\"wasm_js\"",
        )
        .args([
            "build",
            &manifest_dir,
            "--release",
            "--target",
            target,
            "--out-dir",
            &format!("pkg/{}", target),
        ])
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "wasm-pack failed with status {status}"
        )));
    }

    Ok(())
}

fn patch() -> io::Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_err(io::Error::other)?;

    println!("🩹 Applying patch to generate isomorphic code for web and nodejs from {manifest_dir}/src/bin/patch.mjs ...");

    let status = Command::new("node")
        .args([format!("{manifest_dir}/src/bin/patch.mjs")])
        .status()?;

    if !status.success() {
        return Err(io::Error::other(format!(
            "patch.mjs failed with status {status}"
        )));
    }

    Ok(())
}
