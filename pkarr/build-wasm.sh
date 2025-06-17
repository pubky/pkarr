#!/bin/bash
set -e

echo "🚀 Building Pkarr WASM Package..."

# Save the custom package.json if it exists
if [ -f "pkg/package.json" ]; then
    echo "💾 Backing up custom package.json..."
    cp pkg/package.json pkg/package.json.backup
fi

# Build WASM package
echo "🔨 Building WASM..."
wasm-pack build --target nodejs --out-dir pkg --features wasm

# Fix .gitignore content (wasm-pack creates it with "*" but we want "p*")
echo "📝 Fixing .gitignore content..."
echo "pkarr*" > pkg/.gitignore

# Restore custom package.json or enhance the generated one
if [ -f "pkg/package.json.backup" ]; then
    echo "🔄 Restoring custom package.json..."
    mv pkg/package.json.backup pkg/package.json
fi

echo "✅ WASM package built successfully!"
echo "📦 Package location: $(pwd)/pkg"
echo ""
echo "🧪 To run examples:"
echo "  cd pkg && npm run example"
echo "  cd pkg && npm run example:advanced"
echo ""
echo "🔬 To run tests:"
echo "  cd pkg && npm run test                  # Run all test suites"
echo "  cd pkg && npm run test:unit            # Run unit tests only"
echo "  cd pkg && npm run test:integration     # Run integration tests only"
echo "  cd pkg && npm run test:performance     # Run performance benchmarks"
echo "  cd pkg && npm run test:edge-cases      # Run edge case tests"