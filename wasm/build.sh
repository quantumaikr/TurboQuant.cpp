#!/bin/bash
# Build quant.cpp WASM demo
# pthreads for parallel matmul + SIMD + ASYNCIFY for UI streaming
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building quant.cpp WASM (pthreads + SIMD) ==="

if ! command -v emcc &>/dev/null; then
    echo "Error: emcc not found. Install Emscripten SDK."
    exit 1
fi

echo "emcc version: $(emcc --version | head -1)"

emcc "$SCRIPT_DIR/quant_wasm.c" \
    -I"$PROJECT_DIR" \
    -o "$SCRIPT_DIR/quant.js" \
    -O3 \
    -msimd128 \
    -mrelaxed-simd \
    -flto \
    -pthread \
    -s WASM=1 \
    -s INITIAL_MEMORY=256MB \
    -s MAXIMUM_MEMORY=4GB \
    -s ALLOW_MEMORY_GROWTH=1 \
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
for f in quant.js quant.wasm; do
    [ -f "$SCRIPT_DIR/$f" ] && echo "  $f ($(du -h "$SCRIPT_DIR/$f" | cut -f1))"
done
