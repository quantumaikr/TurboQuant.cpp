# quant.cpp Public API Reference

quant.cpp provides two API layers:

1. **Single-Header API (`quant.h`)** -- Drop-in inference with 6 functions. No build system needed.
2. **Full Library API (`libturboquant`)** -- Complete KV cache compression, quantization, and inference control.

---

## Table of Contents

- [Single-Header API (quant.h)](#single-header-api-quanth)
  - [Types](#types)
  - [Functions](#functions)
  - [Complete Example](#complete-example)
- [Full Library API (libturboquant)](#full-library-api-libturboquant)
  - [Context and Backend](#context-and-backend)
  - [Model Loading](#model-loading)
  - [State Management](#state-management)
  - [Inference](#inference)
  - [Generation](#generation)
  - [Tokenizer](#tokenizer)
  - [Quantization Types](#quantization-types)
  - [KV Cache Quantization](#kv-cache-quantization)
  - [Paged Cache Management](#paged-cache-management)
  - [Progressive Compression](#progressive-compression)
  - [Random Hadamard Transform (RHT)](#random-hadamard-transform-rht)
  - [Thread Control](#thread-control)
  - [Tensor Operations](#tensor-operations)
  - [Error Handling](#error-handling)
- [Build Instructions](#build-instructions)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
  - [WebAssembly (Emscripten)](#webassembly-emscripten)
  - [iOS and Android](#ios-and-android)

---

## Single-Header API (quant.h)

The simplest way to use quant.cpp. Include `quant.h` with `QUANT_IMPLEMENTATION` defined in exactly one C file.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"
```

### Types

#### `quant_model`

Opaque handle to a loaded model. Created by `quant_load`, freed by `quant_free_model`.

#### `quant_ctx`

Opaque inference context. Created by `quant_new`, freed by `quant_free_ctx`.

#### `quant_config`

Configuration struct for inference context creation.

```c
typedef struct {
    float temperature;   // Sampling temperature. Default: 0.7
    float top_p;         // Nucleus sampling threshold. Default: 0.9
    int   max_tokens;    // Maximum tokens to generate. Default: 256
    int   n_threads;     // Number of threads for matmul. Default: 4
    int   kv_compress;   // KV cache compression level:
                         //   0 = off (FP32 KV cache)
                         //   1 = 4-bit K+V (default)
                         //   2 = delta + 3-bit compression
} quant_config;
```

### Functions

#### `quant_load`

```c
quant_model* quant_load(const char* path);
```

Load a GGUF model file from disk. Auto-detects format (GGUF v2/v3). Returns `NULL` on failure. The model is memory-mapped for efficient loading.

#### `quant_new`

```c
quant_ctx* quant_new(quant_model* model, const quant_config* config);
```

Create an inference context. Pass `config = NULL` to use defaults (`temperature=0.7`, `top_p=0.9`, `max_tokens=256`, `n_threads=4`, `kv_compress=1`).

#### `quant_generate`

```c
int quant_generate(quant_ctx* ctx, const char* prompt,
                   void (*on_token)(const char* text, void* user_data),
                   void* user_data);
```

Generate tokens from a prompt. Calls `on_token` for each generated token with the decoded text. Returns the number of tokens generated. The callback receives the token text and the opaque `user_data` pointer.

#### `quant_ask`

```c
char* quant_ask(quant_ctx* ctx, const char* prompt);
```

Generate a complete response and return it as a heap-allocated string. The caller must call `free()` on the returned pointer.

#### `quant_free_ctx`

```c
void quant_free_ctx(quant_ctx* ctx);
```

Free the inference context and all associated buffers (KV cache, activation buffers).

#### `quant_free_model`

```c
void quant_free_model(quant_model* model);
```

Free the model and unmap the model file.

#### `quant_version`

```c
const char* quant_version(void);
```

Return the library version string (e.g., `"0.1.0"`).

### Complete Example

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"
#include <stdio.h>

static void print_token(const char* text, void* user_data) {
    (void)user_data;
    printf("%s", text);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }

    // Load model
    quant_model* model = quant_load(argv[1]);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        return 1;
    }

    // Configure: 8 threads, delta+3-bit KV compression
    quant_config config = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .max_tokens = 512,
        .n_threads = 8,
        .kv_compress = 2,
    };

    // Create context
    quant_ctx* ctx = quant_new(model, &config);

    // Generate with streaming
    const char* prompt = (argc > 2) ? argv[2] : "Hello!";
    printf("Prompt: %s\n\n", prompt);
    int n = quant_generate(ctx, prompt, print_token, NULL);
    printf("\n\n[%d tokens generated]\n", n);

    // Or use quant_ask for a complete response
    char* response = quant_ask(ctx, "What is 2+2?");
    if (response) {
        printf("Answer: %s\n", response);
        free(response);
    }

    // Cleanup
    quant_free_ctx(ctx);
    quant_free_model(model);
    return 0;
}
```

Compile with:

```bash
cc -O2 -o chat chat.c -lm -lpthread
```

---

## Full Library API (libturboquant)

The full library provides fine-grained control over model loading, KV cache compression, and inference. Include `<turboquant/turboquant.h>` for quantization functions, and `<turboquant/tq_engine.h>` for inference.

### Context and Backend

#### `tq_init`

```c
tq_status tq_init(tq_context_t** ctx, tq_backend backend);
```

Initialize a TurboQuant context for quantization operations.

| Backend | Value | Description |
|---------|-------|-------------|
| `TQ_BACKEND_CPU` | 0 | CPU reference + SIMD |
| `TQ_BACKEND_CUDA` | 1 | NVIDIA GPU |
| `TQ_BACKEND_METAL` | 2 | Apple GPU |
| `TQ_BACKEND_VULKAN` | 3 | AMD + cross-platform (SPIR-V) |
| `TQ_BACKEND_ROCM` | 4 | AMD ROCm/HIP |
| `TQ_BACKEND_AUTO` | 99 | Auto-detect best backend |

#### `tq_free`

```c
void tq_free(tq_context_t* ctx);
```

Free the context and associated resources.

#### `tq_get_backend`

```c
tq_backend tq_get_backend(const tq_context_t* ctx);
```

Return which backend the context is using.

### Model Loading

#### `tq_load_model`

```c
tq_model_t* tq_load_model(const char* path);
```

Auto-detect file format (GGUF or TQM) and load the model. Uses mmap for zero-copy tensor access. Returns `NULL` on failure.

#### `tq_load_gguf` / `tq_load_tqm`

```c
tq_model_t* tq_load_gguf(const char* path);
tq_model_t* tq_load_tqm(const char* path);
```

Load from a specific format. GGUF supports community models (llama.cpp, Unsloth, bartowski). TQM is the native pre-quantized format.

#### `tq_save_tqm`

```c
int tq_save_tqm(tq_model_t* model, const char* tokenizer_path,
                const char* output_path);
```

Save a loaded model as TQM format for faster subsequent loads.

#### `tq_free_model`

```c
void tq_free_model(tq_model_t* model);
```

Free model weights and unmap the backing file.

### State Management

#### `tq_create_state` / `tq_create_state_ex`

```c
tq_state_t* tq_create_state(const tq_model_config_t* config, tq_type kv_type);
tq_state_t* tq_create_state_ex(const tq_model_config_t* config, tq_type kv_type,
                                int value_quant_bits);
```

Allocate runtime state including activation buffers and KV cache. The `kv_type` parameter selects key quantization (see [Quantization Types](#quantization-types)). Use `tq_create_state_ex` to additionally control value quantization bits (0 = FP32/FP16, 4 = Q4, 2 = Q2).

#### `tq_free_state`

```c
void tq_free_state(tq_state_t* state);
```

Free all state buffers including KV cache.

### Inference

#### `tq_forward`

```c
float* tq_forward(tq_model_t* model, tq_state_t* state, int token, int pos);
```

Run one transformer forward pass. Takes a single token ID and its position in the sequence. Returns a pointer to the logits array (`[vocab_size]` floats, owned by `state`). The KV cache is updated in-place.

### Generation

#### `tq_generate`

```c
int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
                const char* prompt, tq_gen_config_t* config,
                char* output, int output_size);
```

Full text generation pipeline: encode prompt, run forward passes, sample tokens, decode output. Returns the number of tokens generated.

#### `tq_gen_config_t`

```c
typedef struct {
    float temperature;        // Sampling temperature
    float top_p;              // Nucleus sampling threshold
    int   max_tokens;         // Maximum output tokens
    tq_type kv_type;          // Key cache quantization type
    int   value_quant_bits;   // Value cache: 0=FP16/FP32, 4=Q4, 2=Q2
    int   v_highres_window;   // Recent N tokens get FP16 values (0=disabled)
    int   delta_kv;           // 1 = delta KV compression (store key deltas)
    int   delta_iframe_interval; // I-frame interval for delta KV (0=auto=64)
    int   k_highres_window;   // Recent N keys at FP32, rest at 2-bit (0=disabled)
    int   n_threads;          // Thread count for matmul
    float rep_penalty;        // Repetition penalty (1.1 default, 1.0=off)
    int   rep_window;         // Number of recent tokens to penalize (32)
    void (*on_token)(const char* text, void* user_data);
    void* user_data;
} tq_gen_config_t;
```

#### `tq_default_gen_config`

```c
tq_gen_config_t tq_default_gen_config(void);
```

Return a config with sensible defaults.

#### Sampling Functions

```c
int tq_sample_argmax(const float* logits, int vocab_size);
int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p, unsigned long long* rng);
```

`tq_sample_argmax` returns the index of the largest logit (greedy decoding). `tq_sample_topp` applies temperature scaling and top-p nucleus sampling.

### Tokenizer

#### `tq_load_tokenizer`

```c
tq_tokenizer_t* tq_load_tokenizer(const char* path);
tq_tokenizer_t* tq_load_tokenizer_from_memory(const char* data, size_t size);
tq_tokenizer_t* tq_load_tokenizer_from_tqm(const char* tqm_path);
tq_tokenizer_t* tq_load_tokenizer_from_gguf(const void* gguf_ctx);
void            tq_free_tokenizer(tq_tokenizer_t* tok);
```

Load a BPE tokenizer from various sources. The GGUF loader extracts the tokenizer directly from the model file.

#### `tq_encode`

```c
int tq_encode(const tq_tokenizer_t* tok, const char* text,
              int* tokens, int max_tokens, int add_bos);
```

Encode text into token IDs. Set `add_bos = 1` to prepend the beginning-of-sequence token. Returns the number of tokens written.

#### `tq_decode`

```c
const char* tq_decode(const tq_tokenizer_t* tok, int prev_token, int token);
```

Decode a single token ID to text. The `prev_token` is needed for proper handling of leading spaces (sentencepiece convention). Returns a pointer to a static buffer (valid until next call).

### Quantization Types

#### Type Info Functions

```c
const char* tq_type_name(tq_type type);       // e.g., "polar_3b"
float       tq_type_bpe(tq_type type);        // bits per element (with metadata)
size_t      tq_type_block_size(tq_type type);  // elements per block (128 or 256)
size_t      tq_type_type_size(tq_type type);   // bytes per block
int         tq_type_count(void);               // total number of types (13)
tq_type     tq_type_from_name(const char* name); // lookup by name string
```

#### Available Types

| Enum | Name | Bits/Elem | Block Size | Algorithm |
|------|------|-----------|------------|-----------|
| `TQ_TYPE_POLAR_3B` | `polar_3b` | ~4.5 | 128 | PolarQuant (theta:2 + rho:1) |
| `TQ_TYPE_POLAR_4B` | `polar_4b` | ~4.5 | 128 | PolarQuant (theta:2 + rho:2) |
| `TQ_TYPE_QJL_1B` | `qjl_1b` | ~1.2 | 256 | QJL 1-bit sign hash |
| `TQ_TYPE_TURBO_3B` | `turbo_3b` | ~7.5 | 128 | Polar 2b + QJL 1b |
| `TQ_TYPE_TURBO_4B` | `turbo_4b` | ~7.5 | 128 | Polar 3b + QJL 1b |
| `TQ_TYPE_UNIFORM_4B` | `uniform_4b` | ~4.25 | 128 | Min-max 4-bit |
| `TQ_TYPE_UNIFORM_2B` | `uniform_2b` | ~3.0 | 128 | Min-max 2-bit (sub-block) |
| `TQ_TYPE_UNIFORM_3B` | `uniform_3b` | ~4.0 | 128 | Min-max 3-bit (sub-block) |
| `TQ_TYPE_MIXED_4B8` | `mixed_4b8` | ~4.75 | 128 | 4-bit base + FP16 outliers |
| `TQ_TYPE_TURBO_KV_3B` | `turbo_kv_3b` | ~3.5 | 128 | RHT + 2-bit codebook + 1-bit QJL |
| `TQ_TYPE_TURBO_KV_4B` | `turbo_kv_4b` | ~4.5 | 128 | RHT + 3-bit codebook + 1-bit QJL |
| `TQ_TYPE_TURBO_KV_1B` | `turbo_kv_1b` | ~1.5 | 128 | RHT + sign-bit (Hamming) |
| `TQ_TYPE_TURBO_KV_2B` | `turbo_kv_2b` | ~2.5 | 128 | RHT + 1-bit codebook + 1-bit QJL |

#### Strategy Recommendation

```c
tq_type tq_recommend_strategy(int head_dim, int target_bits, float quality_threshold);
```

Given constraints, recommend the best quantization type.

### KV Cache Quantization

#### Quantize Keys

```c
tq_status tq_quantize_keys(tq_context_t* ctx,
                           const float* keys, int n, int head_dim,
                           tq_type type,
                           void* out, size_t out_size);

size_t tq_quantize_keys_size(int n, int head_dim, tq_type type);
```

Quantize `n` key vectors of dimension `head_dim` into the output buffer. Call `tq_quantize_keys_size` first to determine the required buffer size.

#### Quantize Values

```c
tq_status tq_quantize_values(tq_context_t* ctx,
                             const float* values, int n, int head_dim,
                             int bits, void* out, size_t out_size);

size_t tq_quantize_values_size(int n, int head_dim, int bits);
```

Quantize value vectors at the specified bit-width (2 or 4).

#### Asymmetric K/V Quantization

```c
tq_status tq_quantize_kv(tq_context_t* ctx,
                          const float* keys, const float* values,
                          int n, int head_dim,
                          tq_type key_type, tq_type value_type,
                          void* key_out, size_t key_out_size,
                          void* val_out, size_t val_out_size);

size_t tq_quantize_kv_key_size(int n, int head_dim, tq_type key_type);
size_t tq_quantize_kv_value_size(int n, int head_dim, tq_type value_type);
```

Quantize keys and values with independent quantization types. Useful when keys benefit from aggressive compression (e.g., 3-bit) while values need higher precision (e.g., 4-bit).

#### Dequantize Keys

```c
tq_status tq_dequantize_keys(tq_context_t* ctx,
                             const void* quantized, int n, int head_dim,
                             tq_type type, float* out);
```

Dequantize keys back to FP32. Primarily for debugging and testing.

#### Attention from Quantized Cache

```c
tq_status tq_attention(tq_context_t* ctx,
                       const float* query,
                       const void* kv_cache,
                       int seq_len, int head_dim,
                       tq_type type,
                       float* scores);
```

Compute attention scores directly from the quantized KV cache without full dequantization. Each quantization type has its own optimized attention kernel (e.g., QJL uses Hamming distance, Polar uses codebook dot products).

### Paged Cache Management

Block-based KV cache with copy-on-write support for efficient memory sharing across beams.

```c
tq_status tq_cache_create(tq_cache_t** cache, int block_size, int max_blocks,
                          int num_heads, int head_dim, tq_type default_type);

tq_status tq_cache_append(tq_cache_t* cache, int head_idx,
                          const float* key, const float* value, int head_dim);

tq_status tq_cache_get_block(const tq_cache_t* cache, int head_idx, int block_idx,
                             const void** data, tq_type* type);

tq_status tq_cache_get_value(const tq_cache_t* cache, int head_idx, int block_idx,
                             const void** data);

int  tq_cache_seq_len(const tq_cache_t* cache, int head_idx);
void tq_cache_free(tq_cache_t* cache);

// Copy-on-Write operations
tq_status tq_cache_share_block(tq_cache_t* cache, int head_idx, int block_idx);
tq_status tq_cache_free_block(tq_cache_t* cache, int head_idx, int block_idx);
int       tq_cache_block_ref_count(const tq_cache_t* cache, int head_idx, int block_idx);
```

### Progressive Compression

Age-based tiered compression: recent tokens at FP16, warm tokens at 4-bit, cold tokens at lower bit-width.

```c
tq_status tq_progressive_create(tq_progressive_t** out,
                                const tq_progressive_config_t* config,
                                int head_dim, int max_tokens);
tq_status tq_progressive_append(tq_progressive_t* p, const float* key, int head_dim);
tq_status tq_progressive_attention(const tq_progressive_t* p, const float* query,
                                   float* scores, int head_dim);
int       tq_progressive_count(const tq_progressive_t* p);
void      tq_progressive_free(tq_progressive_t* p);

tq_progressive_config_t tq_progressive_default_config(void);
```

#### `tq_progressive_config_t`

```c
typedef struct {
    int      residual_window;     // Tier 0 (FP16) size, default 128
    int      warm_window;         // Tier 1 (4-bit) size, default 256
    tq_type  warm_type;           // Tier 1 quantization type
    tq_type  cold_type;           // Tier 2 quantization type
    int      enable_recompression;// Tier 1 -> Tier 2 re-compression
} tq_progressive_config_t;
```

### Random Hadamard Transform (RHT)

Pre-processing that decorrelates channels for improved quantization quality. Used by TurboKV types internally.

```c
void tq_rht_transform(float* data, int n, uint32_t seed);  // In-place forward RHT
void tq_rht_inverse(float* data, int n, uint32_t seed);    // In-place inverse RHT

// Quantize with RHT pre-processing (higher quality than plain quantize)
tq_status tq_quantize_keys_rht(tq_context_t* ctx,
                                const float* keys, int n, int head_dim,
                                tq_type type, uint32_t rht_seed,
                                void* out, size_t out_size);

tq_status tq_dequantize_keys_rht(tq_context_t* ctx,
                                  const void* quantized, int n, int head_dim,
                                  tq_type type, uint32_t rht_seed,
                                  float* out);
```

### Thread Control

```c
void tq_set_threads(int n_threads);  // Set thread count for matmul
int  tq_get_threads(void);           // Get current thread count
```

Maximum thread pool size is `TQ_TP_MAX` (16).

### Tensor Operations

Low-level operations exported for reuse. All matmul variants support FP32, BF16, Q8, Q4, and Q2 weight formats.

```c
void tq_matmul(float* out, const float* x, const float* w, int n, int d);
void tq_matmul_bf16(float* out, const float* x, const uint16_t* w_bf16, int n, int d);
void tq_matmul_q8(float* out, const float* x, const int8_t* w_qs, const float* w_scales, int n, int d);
void tq_matmul_q4(float* out, const float* x, const uint8_t* w_qs, const float* w_scales, int n, int d);
void tq_matmul_q2(float* out, const float* x, const uint8_t* w_qs, const float* w_scales, int n, int d);

void tq_rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void tq_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads, float freq_base);
void tq_silu(float* x, int n);
void tq_gelu_tanh(float* x, int n);
void tq_softmax(float* x, int n);
void tq_add(float* out, const float* a, const float* b, int n);
void tq_mul(float* out, const float* a, const float* b, int n);
```

### Error Handling

All functions returning `tq_status` use these codes:

```c
typedef enum {
    TQ_OK              =  0,  // Success
    TQ_ERR_NULL_PTR    = -1,  // NULL pointer argument
    TQ_ERR_INVALID_TYPE= -2,  // Unknown quantization type
    TQ_ERR_INVALID_DIM = -3,  // Invalid dimension (not multiple of block size)
    TQ_ERR_OUT_OF_MEM  = -4,  // Memory allocation failed
    TQ_ERR_NOT_IMPL    = -5,  // Feature not implemented
    TQ_ERR_BACKEND     = -6,  // Backend-specific error
    TQ_ERR_BUFFER_TOO_SMALL = -7, // Output buffer too small
} tq_status;

const char* tq_status_string(tq_status status);
```

---

## Build Instructions

### macOS

Default build with Accelerate framework (AMX coprocessor) and Metal GPU:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON -DTQ_BUILD_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

Without Metal (CPU only):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Linux

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

With CUDA (NVIDIA GPU):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_CUDA=ON -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

With Vulkan (AMD GPU or cross-platform):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_VULKAN=ON -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### Windows

#### MSVC (Visual Studio)

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

#### MinGW

```bash
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### WebAssembly (Emscripten)

```bash
emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release
emmake cmake --build build-wasm -j$(nproc)
```

This produces `libturboquant.a` that can be linked into an Emscripten application. SIMD support requires `-msimd128` in CFLAGS.

### iOS and Android

#### iOS (Xcode toolchain)

```bash
cmake -B build-ios \
      -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
      -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_METAL=ON
cmake --build build-ios -j$(sysctl -n hw.ncpu)
```

#### Android (NDK)

```bash
cmake -B build-android \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-26 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j$(nproc)
```

ARM NEON SIMD is auto-detected on arm64 targets.

### Linking

The build produces two artifacts:

- **`libturboquant.a`** -- Static library
- **`libturboquant.so`** (or `.dylib` on macOS) -- Shared library for Python/FFI bindings

Link against the static library:

```bash
cc -O2 my_app.c -Iinclude -Lbuild -lturboquant -lm -lpthread -o my_app
```
