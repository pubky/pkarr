#!/usr/bin/env bash

set -euo pipefail

if ! command -v cross >/dev/null 2>&1; then
    echo "cross is required to build release binaries for multiple targets" >&2
    exit 1
fi

VERSION=$(cargo pkgid -p pkarr-relay | sed 's/.*#//;s/.*@//')
ARTIFACTS_DIR=target/github-release
mkdir -p "$ARTIFACTS_DIR"
rm -f "$ARTIFACTS_DIR"/*.tar.gz

build_target() {
    local target=$1
    local platform=$2
    local archive_name="pkarr-relay-v${VERSION}-${platform}"
    local staging_dir="$ARTIFACTS_DIR/$archive_name"
    # Keep host build scripts separate across images with different libc versions.
    local target_dir="target/cross/$target"
    local binary_name=pkarr-relay
    if [[ "$target" == *-windows-* ]]; then
        binary_name=pkarr-relay.exe
    fi

    echo "Building pkarr-relay for $target"
    CARGO_TARGET_DIR="$target_dir" cross build -p pkarr-relay --release --locked --target "$target"

    mkdir -p "$staging_dir"
    cp "$target_dir/$target/release/$binary_name" "$staging_dir/"
    tar -C "$ARTIFACTS_DIR" -czf "$ARTIFACTS_DIR/$archive_name.tar.gz" "$archive_name"
    rm -rf "$staging_dir"
}

build_target aarch64-unknown-linux-musl linux-arm64
build_target x86_64-unknown-linux-musl linux-amd64
build_target x86_64-pc-windows-gnu windows-amd64
build_target aarch64-apple-darwin osx-arm64
build_target x86_64-apple-darwin osx-amd64

ls -lh "$ARTIFACTS_DIR"
