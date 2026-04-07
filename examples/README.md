# quant.cpp Embedding Examples

This directory contains examples demonstrating how to embed quant.cpp (the SQLite of LLM inference) into your C/C++ projects.

## Quick Start

The simplest way to use quant.cpp is with the single-header `quant.h`. No build system required:

```bash
cc -O2 -o chat embed_chat.c -lm -lpthread
./chat model.gguf
```

## Examples

### embed_minimal.c
**The smallest possible LLM integration (~60 lines)**

Demonstrates the 6-function API:
- `quant_load()` - Load GGUF model
- `quant_new()` - Create inference context
- `quant_generate()` - Stream tokens via callback
- `quant_free_ctx()` / `quant_free_model()` - Cleanup

**Compile:**
```bash
cc -O2 embed_minimal.c -o minimal -lm -lpthread
```

**Run:**
```bash
./minimal smollm2-135m.gguf "Tell me a joke"
```

**Features:**
- Zero dependencies (libc + pthreads)
- Memory-mapped model loading
- KV cache compression enabled (7x longer context on same hardware)
- Streaming token output

---

### embed_chat.c
**Interactive chat application (~60 lines)**

A complete REPL (Read-Eval-Print Loop) for conversational AI.

**Compile:**
```bash
cc -O2 embed_chat.c -o chat -lm -lpthread
```

**Run:**
```bash
./chat model.gguf
```

**Features:**
- Interactive prompt loop
- Fresh context per turn (no conversation memory)
- Ctrl+C to exit
- Streaming output

**Usage:**
```
Model loaded. Type your message (Ctrl+C to exit).

> Hello!
[AI response streaming...]
```

---

### embed_kv_compare.c
**KV compression quality comparison (~60 lines)**

Runs the same prompt with different KV compression levels to demonstrate quality vs. memory trade-offs.

**Compile:**
```bash
cc -O2 embed_kv_compare.c -o kv_compare -lm -lpthread
```

**Run:**
```bash
./kv_compare model.gguf
```

**Output:**
```
Prompt: What is the capital of France?
==========================================

[FP32 (no compression)]
  Output: Paris

[4-bit K + Q4 V]
  Output: Paris

[Delta 3-bit + Q4 V]
  Output: Paris
```

**Compression Levels:**
- `kv_compress=0` - FP32 KV cache (no compression, highest quality)
- `kv_compress=1` - 4-bit K + Q4 V (7x compression, PPL +0.0%)
- `kv_compress=2` - Delta 3-bit + Q4 V (aggressive compression)

---

### single_header_example.c
**Minimal single-header example (~40 lines)**

The absolute minimum code needed to run inference.

**Compile:**
```bash
cc -O2 single_header_example.c -o example -lm -lpthread
```

**Run:**
```bash
./example model.gguf "Hello, world!"
```

---

## Platform Support

All examples work on:
- **macOS** (Apple Silicon, Intel)
- **Linux** (x86_64, ARM64)
- **Windows** (MSVC, MinGW)
- **WebAssembly** (via Emscripten)
- **iOS** (Xcode toolchain)
- **Android** (NDK)

No external dependencies required beyond libc and pthreads.

## quant.h API Reference

### Configuration

```c
typedef struct {
    float temperature;   // Sampling temperature (default: 0.7)
    float top_p;         // Nucleus sampling (default: 0.9)
    int   max_tokens;    // Max tokens to generate (default: 256)
    int   n_threads;     // Thread count for matmul (default: 4)
    int   kv_compress;   // 0=off, 1=4-bit, 2=delta+3-bit (default: 1)
} quant_config;
```

### Functions

| Function | Description |
|----------|-------------|
| `quant_load(path)` | Load GGUF model from disk |
| `quant_new(model, config)` | Create inference context |
| `quant_generate(ctx, prompt, cb, ud)` | Stream tokens via callback |
| `quant_ask(ctx, prompt)` | Return full response (caller frees) |
| `quant_free_ctx(ctx)` | Free context |
| `quant_free_model(model)` | Free model |
| `quant_version()` | Get version string |

## Memory Requirements

- **Model loading**: Memory-mapped, minimal RAM overhead
- **Inference context**: ~2-4GB for 7B models (depends on sequence length)
- **KV cache compression**: 7x reduction vs FP32 baseline

## Performance Tips

1. **Use KV compression** (`kv_compress=1` or `2`) for 7x longer context
2. **Adjust thread count** (`n_threads`) based on CPU cores
3. **Lower temperature** (0.0-0.3) for factual responses
4. **Higher temperature** (0.7-1.0) for creative writing

## Troubleshooting

**"Failed to load model"**
- Check model path is correct
- Verify model is in GGUF format (llama.cpp compatible)
- Ensure sufficient disk space for mmap

**Slow generation**
- Increase `n_threads` (up to CPU core count)
- Enable KV compression if not already
- Try smaller models for faster inference

**Poor quality output**
- Adjust `temperature` and `top_p` parameters
- Try different KV compression levels
- Ensure model is appropriate for your use case

## Next Steps

- See `docs/api.md` for full API documentation
- Check `examples/` for advanced usage patterns
- Read `README.md` for project overview
- Visit https://github.com/quantumaikr/quant.cpp for more information

## License

Apache 2.0 - See LICENSE file for details.
