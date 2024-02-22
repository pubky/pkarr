#!/bin/bash
 
# Build the dylib
cargo build --release
 
# Generate bindings
cargo run --bin uniffi-bindgen generate --library ./target/release/libpkarrbindings.dylib --language swift --out-dir ./ffi/bindings || exit 1
 
for TARGET in \
        aarch64-apple-darwin \
        aarch64-apple-ios \
        aarch64-apple-ios-sim \
        x86_64-apple-darwin \
        x86_64-apple-ios
do
    rustup target add $TARGET  || exit 1
    cargo build --release --target=$TARGET  || exit 1
    echo "\nBuilt for $TARGET"
done
 
mv ./ffi/bindings/pkarrbindingsFFI.modulemap ./ffi/bindings/module.modulemap
 
# Create XCFramework
rm -rf "./ffi/swift/Pkarr.xcframework"
xcodebuild -create-xcframework \
        -library ./target/aarch64-apple-ios-sim/release/libpkarrbindings.a -headers ./ffi/bindings \
        -library ./target/aarch64-apple-ios/release/libpkarrbindings.a -headers ./ffi/bindings \
        -output "./ffi/swift/Pkarr.xcframework"
 
