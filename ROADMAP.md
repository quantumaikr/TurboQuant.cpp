# quant.cpp Roadmap

## Vision

**quant.cpp is the SQLite of LLM inference.**

Not the fastest. Not the most feature-complete.
The most embeddable, the most readable, and the only engine
that compresses KV cache 7x without quality loss.

## Positioning

```
Need speed?        → llama.cpp
Need throughput?   → vLLM
Need to embed LLM in your app with one file? → quant.cpp
Need 7x longer context on the same hardware? → quant.cpp
```

## Direction 1: Embedding Engine ("LLM의 SQLite")

The world's simplest way to add LLM to a C/C++ project.

### Done
- [x] quant.h single header (15K LOC, 628KB)
- [x] 6-function API (load, new, generate, ask, free_ctx, free_model)
- [x] WASM build (192KB binary)
- [x] MSVC/MinGW Windows support
- [x] Zero external dependencies
- [x] API documentation (docs/api.md)
- [x] quant.h sync with latest source
- [x] Embedding examples (minimal, chat, KV compare)

### Planned
- [ ] pip install quantcpp (Python bindings)
- [ ] iOS SDK + demo app
- [ ] Android NDK build guide
- [ ] Unity C# plugin
- [ ] Unreal C++ integration
- [ ] npm package (WASM)
- [ ] GitHub Pages live demo with pre-loaded model

## Direction 2: KV Compression Research Platform

The reference implementation for KV cache quantization research.

### Done
- [x] 7 quantization types (Polar, QJL, Turbo, Uniform, TurboKV)
- [x] Delta compression (P-frame encoding)
- [x] QK-norm aware compression
- [x] Plugin architecture (3 functions to add new type)
- [x] 34 unit tests

### In Progress
- [ ] "Add Your Own Type" tutorial (docs/custom-quantization.md)
- [ ] Arxiv tech report

### Planned
- [ ] llama.cpp KV type PR (ggml type registration)
- [ ] vLLM KV compression plugin
- [ ] Benchmarking suite (PPL across models × KV types)
- [ ] Learned codebook quantization
- [ ] Per-head adaptive bit allocation

## Non-Goals

- ❌ GPU speed competition with llama.cpp (requires tensor graph IR)
- ❌ Batch serving (vLLM's domain)
- ❌ Training support
- ❌ 100+ model coverage

## Architecture Principles

1. **One file forward pass**: tq_transformer.c contains the entire inference loop
2. **Plugin quantization**: Add types via tq_traits.c registration
3. **Zero dependencies**: libc + pthreads only (+ Metal on macOS)
4. **CPU-first**: NEON/AVX2 optimized, GPU as optional accelerator
5. **Embeddable**: quant.h works anywhere a C compiler does
