# quant.cpp Roadmap

## Vision

**quant.cpp is the single-header C reference implementation of TurboQuant and related KV cache quantization research.**

Not competing with Google. Not competing with llama.cpp.
Filling the gap nobody else fills: TurboQuant-class compression *anywhere* a C compiler runs.

See [docs/positioning.md](docs/positioning.md) for the full strategy.

## Positioning

```
Data-center TurboQuant?       → Google reference (arxiv:2504.19874)
Workstation speed?            → llama.cpp
Batch serving?                → vLLM
TurboQuant on iPhone?         → quant.cpp
TurboQuant in a browser?      → quant.cpp
TurboQuant in a game engine?  → quant.cpp
TurboQuant on a microcontroller? → quant.cpp
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

A C reference engine for KV cache quantization research.

### Production-ready
- [x] **`turbo_kv_4b` ⭐** — RHT + 4-bit Lloyd-Max codebook, beats `uniform_4b` and llama.cpp `q4_0` KV at the same bit budget (Llama 3.2 3B PPL 14.28, +5.3% vs FP32)
- [x] **`turbo_kv_3b`** — RHT + 3-bit Lloyd-Max codebook (PPL 15.39, +13.5%)
- [x] `uniform_4b` KV quantization (4–7x compression, +6.3% PPL on Llama 3.2 3B)
- [x] `uniform_4b` + Q4 V combo (6.9x KV memory reduction)
- [x] Delta compression (P-frame encoding)
- [x] QK-norm aware compression (Gemma 4 / hybrid attention models)
- [x] Plugin architecture (3 functions to add new type)
- [x] 35 unit tests

### Building blocks
- [x] Random Hadamard Transform (`tq_rht.c`)
- [x] Lloyd-Max-Gaussian codebook quantizer (`tq_codebook.c`, 1–4 bit)
- [x] 1-bit QJL sign hash (`tq_qjl.c`) — research, contributes ~0 to scores in our regime
- [x] PolarQuant (polar coordinate) compression (`tq_polar.c`)

### TurboQuant paper reproduction (issue #14, partially resolved)
- [x] Identify the gap in literal port (commit 4da6915 — QJL contributes byte-identical zero)
- [x] Variant F: drop QJL stage, double codebook size (commit ac3c46a — beats baseline)
- [ ] Per-channel outlier handling (Google paper's 32-channel split)
- [ ] Paper-faithful Llama 3.1 8B + LongBench-E reproduction
- [ ] 5-bit codebook variant for ~5 bpc quality budget

### Planned (after Direction 2 reproduction)
- [ ] "Add Your Own Type" tutorial polish (docs/custom-quantization.md)
- [ ] Arxiv tech report
- [ ] llama.cpp KV type PR (ggml type registration) — only after paper reproduction works
- [ ] vLLM KV compression plugin
- [ ] Benchmarking suite (PPL across models × KV types)

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
