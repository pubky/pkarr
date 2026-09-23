//! The relay binary must be able to report which version it is running.

use std::process::Command;

#[test]
fn version_flag_prints_the_crate_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_pkarr-relay"))
        .arg("--version")
        .output()
        .expect("pkarr-relay binary should be runnable");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "`pkarr-relay --version` exited with {}\nstdout: {stdout}\nstderr: {stderr}",
        output.status
    );

    assert_eq!(
        stdout.trim(),
        format!("pkarr-relay {}", env!("CARGO_PKG_VERSION")),
        "`pkarr-relay --version` should print the crate version"
    );
}
