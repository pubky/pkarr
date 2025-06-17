#!/bin/bash

# Build script for pkarr WASM bindings

set -e

echo "🚀 Building pkarr WASM bindings..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed. Please install it with:"
    echo "   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Navigate to pkarr directory
cd pkarr

# Build the WASM package
echo "📦 Building WASM package..."
wasm-pack build \
    --target web \
    --features wasm

# Move the pkg directory to the parent directory
if [ -d "pkg" ]; then
    echo "📁 Moving pkg directory to parent..."
    mv pkg ../pkg
fi

echo "✅ WASM build complete!"
echo "📁 Output directory: ./pkg"
echo ""
echo "🎯 Usage example:"
echo "   import init, { WasmPkarrClient, WasmKeypair, WasmSignedPacketBuilder } from './pkg/pkarr.js';"
echo ""
echo "📚 Next steps:"
echo "   1. Copy the ./pkg directory to your web project"
echo "   2. Import and use the WASM module in your JavaScript/TypeScript code"
echo "   3. See the generated TypeScript definitions in ./pkg/pkarr.d.ts" 