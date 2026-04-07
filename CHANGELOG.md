# Changelog

## [0.6.0] — 2026-04-08

### Highlights

- **🏆 turbo_kv_4b is the new champion** — Beats both `uniform_4b` and llama.cpp `q4_0` KV at the same 4-bit budget on Llama 3.2 3B (PPL 14.28 vs 14.41 vs ~14.99). Reached after 6 rounds of Karpathy-loop iteration starting from a literal port of [Google TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874).
- **CLI default switched** — `quant model.gguf` now uses `turbo_kv_4b` automatically. `uniform_4b` remains available via `-k uniform_4b`.
- **Honest TurboQuant reproduction story** — full ablation history, public issue #14, no overstated claims. The shipped `turbo_kv_*` is structurally simpler than the paper (single-stage RHT + Lloyd-Max codebook + ‖x‖) but empirically beats the literal two-stage port on our benchmark.
- **@quantcpp/wasm npm package** — `npm install @quantcpp/wasm` to drop a 192KB GGUF inference engine into any web project.
- **Windows CI green** — pthread_cond_wait SRWLOCK deadlock fixed, MSVC `__builtin_*` shims, /tmp paths in tests, M_PI in test_neon_scalar. 35/35 tests pass on macOS / Linux / Windows.
- **Public PR & issue triage** — PR #12 (5 critical bug fixes from MChorfa) cherry-picked into main; PR #13 reformatting noise rejected, examples README + CMake separation salvaged.

### KV quantization

The `turbo_kv_3b` / `turbo_kv_4b` block layouts changed in this release. The `qjl_signs` field is gone — Karpathy-loop ablation showed it contributed byte-identical zero to attention scores. The freed 16 bytes per block are now used for a 2× larger Lloyd-Max codebook. Same total block size, finer reconstruction, single-stage estimator.

| KV type | Bits/elem | Llama 3.2 3B PPL | Δ vs FP32 |
|---|---:|---:|---:|
| FP32 baseline | 32 | 13.56 | — |
| **`turbo_kv_4b`** ⭐ | 4 | **14.28** | **+5.3%** |
| `uniform_4b` | 4 | 14.41 | +6.3% |
| `turbo_kv_3b` | 3 | 15.39 | +13.5% |
| llama.cpp q4_0 KV (rough) | 4 | ~14.99 | +10.6% |

### Strategy & positioning

- New `docs/positioning.md` — quant.cpp = the single-header C reference engine for the embedded niche (iOS, Android, WASM, MSVC, microcontrollers, game engines)
- README repositioned to honest "production = turbo_kv_4b, research = building blocks" framing with full PPL methodology
- Citations to Google TurboQuant, PolarQuant, QJL papers added throughout

### Tooling & ecosystem

- `wasm/package.json` + ESM `index.mjs` + `index.d.ts` for npm publishing
- `examples/README.md` (cherry-picked from PR #13) — comprehensive embedding examples doc
- CMake `TQ_BUILD_EXAMPLES` option, single-header examples link only against libm + threads
- Windows CI test timeouts bumped to 600s for slow non-vectorized builds

### Bug fixes (cherry-picked from PR #12)

- `tq_qjl.c`: NaN guard requires `dim > 0`
- `tq_uniform.c`: heap-allocate Q8 query buffer (was 512B stack array)
- `tq_transformer.c`: NULL-check key/value cache calloc results
- `tq_ops.c`: Windows pthread_cond_wait must use `SleepConditionVariableSRW` not `CS` (caused test_ops deadlock on first Windows green run)

### Tracked for next release (issue #14 follow-ups)

- Per-channel outlier handling (Google paper's 32-channel split)
- Paper-faithful Llama 3.1 8B + LongBench-E reproduction
- 5-bit codebook variant for higher quality at ~5 bpc

## [0.5.0] — 2026-04-05

### Highlights

- **Gemma 4 26B-A4B MoE** — Full support for 128-expert MoE, dual-FFN, hybrid attention, QK-norm, learned RoPE, GeGLU
- **Llama 3.2 3B** — Verified at 17 tok/s with correct code generation
- **7x KV cache compression** — 350K tokens on 16GB Mac (was 50K), PPL +0.0%
- **quant.h synced** — Single header now includes IQ3_XXS, Gemma 4, Llama 3 support
- **WASM browser demo** — 192KB binary, GitHub Pages deployed
- **MSVC Windows** — Visual Studio 2019/2022 compilation support
- **Metal GPU infrastructure** — 7 compute kernels, full-layer pipeline (disabled for batch-1, ready for batch inference)

### Architecture

- Gemma 4 MoE: 10 architecture bugs fixed (dual-FFN loading, layer_output_scale, attention_scale, V-norm, etc.)
- QK-norm aware KV compression: auto FP32 keys for sparse distributions (cosine 0.62 → 1.00)
- IQ3_XXS dequantization with 256-entry grid codebook
- NEON fused dot for IQ3_XXS, IQ4_NL, Q8_0 (two-accumulator with prefetch)
- GeGLU NEON (fast tanh via Schraudolph approximation)
- GGUF embedding: skip 2.8GB FP32 allocation, use Q6_K fused dot directly

### Performance

- Gemma 4 26B: 549ms → 257ms/token (-53%)
- SmolLM2 135M: 96 tok/s (CPU NEON Q4×Q8)
- Llama 3.2 3B: 17 tok/s
- KV compression: uniform_4b + Q4V = 6.9x, delta 3b + Q4V = 8.5x

### Platform

- MSVC: pthread shim (CRITICAL_SECTION), _Interlocked atomics, QueryPerformanceCounter
- WASM: Emscripten build (192KB), drag-and-drop GGUF, streaming output
- Metal: RoPE, GELU, softmax, attention_qk, attention_v, kv_cache_write, matmul_tq_q4_fast kernels
- GitHub Pages: live demo at quantumaikr.github.io/quant.cpp

### Documentation

- docs/api.md: Full C API reference (730 lines) — single-header + full library
- docs/custom-quantization.md: Step-by-step guide to add new KV quantization types
- docs/papers/quant_cpp_tech_report.md: Arxiv tech report draft
- ROADMAP.md: Project direction ("LLM의 SQLite" + research platform)
- 3 embedding examples: minimal (30 lines), chat (60 lines), KV compare

### Bug Fixes

- Fix Gemma 4 NaN regression (model_type set after hybrid detection)
- Fix Llama GQA head_dim misdetection (was 64 instead of 128)
- Fix TQ_STATIC_ASSERT no-op in C mode (now C11 _Static_assert)
- Fix stack buffer overflow in attention debug (recon[256] → recon[512])
- Fix Accelerate macro redefinition warning

### Community

- llama.cpp Discussion #20969: shared TurboQuant implementation status
- vLLM-omni Issue #2215: KV compression RFC contribution
- Closed Issues #5 (Q3_K_M), #7 (API docs), #10 (MSVC), #11 (static_assert)

---

## [0.1.0] — 2026-03-29

### Highlights

- **Self-contained inference engine** — loads Qwen3.5-0.8B, generates text at 14 tok/s on CPU
- **17x faster than PyTorch CPU**, 1.4x faster than PyTorch on Apple GPU
- **Q8 weight quantization** — 4x memory reduction (2.1 GB → 533 MB), `-q` flag
- **Streaming BF16** — embed/lm_head kept as mmap'd BF16, saves ~1 GB
- **Multi-threaded matmul** — 4-thread pthread, 1.56x speedup
- **DeltaNet + Self-Attention** — full Qwen3.5 hybrid architecture in C
- **HuggingFace BPE tokenizer** — 248K vocab, encode/decode
- **KV cache quantized in inference** — Q4 keys, integer Q4×Q8 attention

### v0.8 Inference Engine

- **Integer-domain attention**: 2.9-4.8x faster than FP32 on Apple Silicon (ARM NEON `vdotq_s32`)
- **Real model validated**: Qwen3.5-0.8B KV cache, cosine 0.994 (A+)
- **8 quantization types** including mixed precision outlier and RHT pre-rotation
- **K/V asymmetric**: independent key/value bit allocation (K4V2 = 9.8x compression)
- **Community validated**: r/LocalLLaMA findings integrated

### Integer-Domain Attention (v0.7)

The single biggest performance breakthrough: instead of dequantizing Q4 keys to FP32,
quantize the query to Q8 and compute integer dot products directly.

```
Before (v0.6): Q4 key → dequantize → FP32 dot = 0.49x vs FP32 (SLOWER)
After  (v0.7): Q4 key × Q8 query → integer dot = 2.9-4.8x vs FP32 (FASTER)
```

Fair NEON-vs-NEON benchmark (Apple M-series, median of 7 runs):
- dim=128, seq=2048: FP32 22.8μs → Int Q4×Q8 7.8μs (2.9x)
- dim=256, seq=2048: FP32 57.7μs → Int Q4×Q8 12.5μs (4.6x)
- Larger head_dim benefits more (Q4 data fits in L1 cache)

### Core Library
- 7 quantization types: PolarQuant (3/4b), QJL (1b), quant.cpp (3/4b), Uniform (2/4b)
- Direct attention kernels: QJL Hamming distance, PolarQuant cos/sin LUT (no dequantization needed)
- Self-contained block formats with ONNX-compliant LSB-first bit packing
- O(1) type traits dispatch table (llama.cpp pattern)
- Thread-safe API with pthread mutex (TSan verified)
- Cross-platform math constants (TQ_PI/TQ_PI_2, no M_PI dependency)

### Cache Management
- Paged KV cache with block-table mapping (vLLM pattern)
- Progressive compression: 3-tier automatic degradation by age, O(1) append
- Copy-on-Write for beam search (ref_count based)
- Value cache quantization and retrieval

### Backends
- CPU Generic (reference C11, zero external dependencies)
- ARM NEON optimized (5.74x speedup over generic)
- x86 AVX2 stubs ready for implementation
- CUDA kernels: 7 files (polar, qjl, turbo, fused_cache, value, common, dispatch)
- Metal compute shaders: 7 files (polar, qjl, turbo, fused_cache, value, common, dispatch)

### Validation
- **A/B test**: uniform_4b achieves cosine 0.995 vs FP16 — A+ grade, virtually lossless
- **Real model validation**: cosine 0.991 on Qwen3.5-0.5B KV cache patterns (4 layers, 14 heads)
- Per-layer analysis: quality consistent across depth (cosine >0.98 for uniform_4b)
- Roundtrip MSE: 0.0014 (synthetic), 0.0025 (real model data)

### Performance (Apple M-series ARM)
- Quantize throughput: 1.4M elements/ms
- Attention throughput: 137K queries/sec
- Compression ratio: 7.53x (uniform_4b)
- SIMD speedup: 4.0x (NEON vs generic)

### Testing
- 13 C++ test suites (Google Test): polar, qjl, turbo, uniform, value, paged_cache, progressive, simd_neon, simd_avx2, threading, edge_cases, attention_all_types, llamacpp_integration
- 22 Python tests (unittest): bindings, roundtrip, attention, types
- Total: **35 tests, 100% pass rate**
- Sanitizers: ASan + UBSan + TSan clean

### Integration
- **llama.cpp**: GGML type registration (7 types, base offset 256), CLI parser with 21 aliases, from_float/to_float/vec_dot wrappers, 10 integration tests
- **Python**: ctypes bindings with NumPy support, pip installable (`pip install -e .`), quant.cpp class with quantize_keys/dequantize_keys/attention methods
- **vLLM**: integration scaffold with README guide
- **Examples**: minimal.c (10 lines), standalone.c, ab_test.c, demo_real_model.c, benchmark_types.cpp, python_quickstart.py, llamacpp_integration.cpp

### Production Readiness (v0.4)
- Integer overflow protection in size calculations
- NULL pointer and buffer size validation on all public APIs
- Edge case defense: seq_len=0, head_dim<2, odd dimensions
- TQ_ERR_BUFFER_TOO_SMALL error code
- tq_type_from_name() / tq_type_count() convenience functions
- BPE values computed from actual struct sizes

### Developer Experience
- 5-dimension scoring harness: structure/correctness/quality/performance/integration
- Hierarchical Harness methodology (Karpathy AutoResearch + ClawTeam multi-agent)
- Agent definitions (.claude/agents/): architect, core-dev, perf-dev, qa
- Skill definitions (.claude/skills/): orchestrate, develop, score, qa
- Slash commands (.claude/commands/): /score, /develop, /harness, /spawn-team, /merge-gate
- PRD documents: v0.1 through v0.4
- WBS documents: v0.1 through v0.4
- refs/ absorption audit with checklist

### Memory Impact

| Model | Context | FP16 Cache | quant.cpp | Saved |
|-------|---------|------------|------------|-------|
| Llama-3.2-3B | 64K | 7.00 GB | 0.93 GB | **87%** |
| Qwen3.5-0.5B | 128K | 10.50 GB | 2.79 GB | **73%** |
| Phi-3-mini | 16K | 6.00 GB | 1.59 GB | **73%** |

### References
- quant.cpp (ICLR 2026) — arXiv:2504.19874
- QJL (AAAI 2025) — arXiv:2406.03482
- PolarQuant (AISTATS 2026) — arXiv:2502.02617
- Harness plugin (revfactory/harness) — agent team methodology
