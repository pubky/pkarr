#!/bin/bash
set -e

echo "🚀 Building Pkarr WASM Package..."

# Save the custom package.json if it exists
if [ -f "pkg/package.json" ]; then
    echo "💾 Backing up custom package.json..."
    cp pkg/package.json pkg/package.json.backup
fi

# Build the WASM package for JavaScript bindings
echo "🔨 Building WASM for JavaScript bindings…"
wasm-pack build --release --target nodejs --out-dir pkg --out-name pkarr

echo ""
# Fix .gitignore content (wasm-pack creates it with "*" but we want "p*")
echo "📝 Fixing .gitignore content..."
echo "pkarr*" > pkg/.gitignore

# Copy README from root to pkg
# echo "📖 Copying README.md to package..."
# cp README.md pkg/README.md

# Restore custom package.json or enhance the generated one
if [ -f "pkg/package.json.backup" ]; then
    echo "🔄 Restoring custom package.json..."
    mv pkg/package.json.backup pkg/package.json
fi

echo "✅ WASM package built successfully!"
echo "📦 Package location: $(pwd)/pkg"