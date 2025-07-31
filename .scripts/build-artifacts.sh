#!/bin/bash

# -------------------------------------------------------------------------------------------------
# This script prepares the artifacts for the current project.
# It builds all the binaries and prepares them for upload as a Github Release.
# The end result will be a target/github-release directory with the following structure:
#
# target/github-release/
# ├── pkarr-relay-v0.5.0-rc.0-linux-arm64.tar.gz
# ├── pkarr-relay-v0.5.0-rc.0-linux-amd64.tar.gz
# ├── pkarr-relay-v0.5.0-rc.0-windows-amd64.tar.gz
# ├── pkarr-relay-v0.5.0-rc.0-osx-arm64.tar.gz
# ├── pkarr-relay-v0.5.0-rc.0-osx-amd64.tar.gz
# └── ...
#
# Make sure you installed https://github.com/cross-rs/cross for cross-compilation.
# To build MacOS, you need access to the [Pubky Github Packages](https://github.com/orgs/pubky/packages).
# -------------------------------------------------------------------------------------------------


set -e # fail the script if any command fails
set -u # fail the script if any variable is not set
set -o pipefail # fail the script if any pipe command fails


# Check if cross is installed
if ! command -v cross &> /dev/null
then
    echo "cross executable could not be found. It is required to cross-compile the binaries. Please install it from https://github.com/cross-rs/cross"
    exit 1
fi

# Read the version from the homeserver
VERSION=$(cargo pkgid -p pkarr-relay | awk -F# '{print $NF}')
echo "Preparing release executables for version $VERSION..."
TARGETS=(
# target, nickname
"aarch64-unknown-linux-musl,linux-arm64"
"x86_64-unknown-linux-musl,linux-amd64"
"x86_64-pc-windows-gnu,windows-amd64"
# "aarch64-apple-darwin,osx-arm64" 
# "x86_64-apple-darwin,osx-amd64"
)

# List of binaries to build.
ARTIFACTS=("pkarr-relay")

echo "Create the github-release directory..."
rm -rf target/github-release
mkdir -p target/github-release

# Helper function to build an artifact for one specific target.
build_target() {
    local TARGET=$1
    local NICKNAME=$2
    echo "Build $NICKNAME with $TARGET"
    FOLDER="pkarr-relay-v$VERSION-$NICKNAME"
    DICT="target/github-release/$FOLDER"
    mkdir -p $DICT
    for ARTIFACT in "${ARTIFACTS[@]}"; do
        echo "- Build $ARTIFACT with $TARGET"
        cross build -p $ARTIFACT --release --target $TARGET
        if [[ $TARGET == *"windows"* ]]; then
            cp target/$TARGET/release/$ARTIFACT.exe $DICT/
        else
            cp target/$TARGET/release/$ARTIFACT $DICT/
        fi
        echo "[Done] Artifact $ARTIFACT built for $TARGET"
    done;
    (cd target/github-release && tar -czf $FOLDER.tar.gz $FOLDER && rm -rf $FOLDER)
}

# Build the binaries
echo "Build all the binaries for version $VERSION..."
for ELEMENT in "${TARGETS[@]}"; do
    # Split tuple by comma
    IFS=',' read -r TARGET NICKNAME <<< "$ELEMENT"

    build_target $TARGET $NICKNAME
done

tree target/github-release
(cd target/github-release && pwd)
