#!/bin/bash
# Build quant.cpp WASM demo (multi-threaded + SIMD)
# Requires: Emscripten SDK (emcc)
#
# Usage: cd wasm && bash build.sh
# Then:  python3 -m http.server 8080
# Open:  http://localhost:8080
#
# Multi-threading requires Cross-Origin-Isolation headers.
# coi-serviceworker.js injects them on GitHub Pages / static hosts.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building quant.cpp WASM (pthreads + SIMD) ==="

# Check emcc
if ! command -v emcc &>/dev/null; then
    echo "Error: emcc not found. Install Emscripten:"
    echo "  brew install emscripten"
    echo "  # or: git clone https://github.com/emscripten-core/emsdk && ./emsdk install latest && ./emsdk activate latest"
    exit 1
fi

echo "emcc version: $(emcc --version | head -1)"

# Build with pthreads + SIMD128 + ASYNCIFY
emcc "$SCRIPT_DIR/quant_wasm.c" \
    -I"$PROJECT_DIR" \
    -o "$SCRIPT_DIR/quant.js" \
    -O3 \
    -msimd128 \
    -flto \
    -pthread \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=4GB \
    -s INITIAL_MEMORY=256MB \
    -s EXPORTED_FUNCTIONS='["_main","_wasm_load_model","_wasm_generate","_wasm_generate_async","_wasm_model_info","_wasm_is_ready","_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["UTF8ToString","allocateUTF8","FS"]' \
    -s FORCE_FILESYSTEM=1 \
    -s MODULARIZE=0 \
    -s ENVIRONMENT='web,worker' \
    -s NO_EXIT_RUNTIME=1 \
    -s ASSERTIONS=0 \
    -s STACK_SIZE=1MB \
    -s ASYNCIFY \
    -s 'ASYNCIFY_IMPORTS=["emscripten_sleep"]' \
    -s ASYNCIFY_STACK_SIZE=65536 \
    -s PTHREAD_POOL_SIZE=4 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -lm \
    -DNDEBUG \
    -D__EMSCRIPTEN__ \
    -Wno-gnu-zero-variadic-macro-arguments \
    -Wno-dollar-in-identifier-extension

echo ""
echo "=== Build complete ==="
echo "Files:"
for f in quant.js quant.wasm quant.worker.js; do
    [ -f "$SCRIPT_DIR/$f" ] && echo "  $f ($(du -h "$SCRIPT_DIR/$f" | cut -f1))"
done
echo ""
echo "To serve locally:"
echo "  cd $SCRIPT_DIR && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
echo ""
echo "Note: Multi-threading requires Cross-Origin-Isolation."
echo "coi-serviceworker.js handles this automatically on GitHub Pages."
