#!/usr/bin/env bash
set -euo pipefail

# Cargo invokes this script as RUSTC_WRAPPER with rustc and its arguments.
# Change only workspace crates: external dependencies may require manifests.
if (( $# > 0 )); then
    previous_argument=""
    for argument in "$@"; do
        if [[ "$previous_argument" == "--crate-name" ]]; then
            case "$argument" in
                pkarr|pkarr_relay)
                    export CARGO_MANIFEST_DIR="$PKARR_TEST_MANIFEST_DIR"
                    echo "Compiling $argument with no Cargo manifest in CARGO_MANIFEST_DIR" >&2
                    break
                    ;;
            esac
        fi
        previous_argument="$argument"
    done
    exec "$@"
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_directory="$(cd -- "$script_directory/.." && pwd)"
test_directory="$(mktemp -d)"
trap 'rm -rf -- "$test_directory"' EXIT

mkdir -- "$test_directory/manifest"
export PKARR_TEST_MANIFEST_DIR="$test_directory/manifest"
export RUSTC_WRAPPER="$script_directory/test-without-cargo-manifest.sh"

# Use a fresh target directory so cached artifacts cannot bypass the check.
export CARGO_TARGET_DIR="$test_directory/target"
cd -- "$workspace_directory"

# Cargo prepares dependencies and build-script flags. The wrapper then invokes
# rustc for workspace crates with an empty manifest directory, as in GN builds.
# The documentation macro is unconditional with respect to crate features, so
# the minimal library build is sufficient to exercise this regression.
cargo rustc -p pkarr --lib --no-default-features --locked

# Cover the relay library and binary, including their PKARR dependency. Retain
# package metadata such as CARGO_PKG_VERSION, which other build systems must
# supply for version reporting; only the manifest directory is unavailable.
cargo build -p pkarr-relay --lib --bins --all-features --locked
