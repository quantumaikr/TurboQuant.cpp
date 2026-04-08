# Changelog

## [0.6.3] — 2026-04-08

### Karpathy round 5+6: closes turbo_kv speed gap from −45% to −8%

> **Correction**: this entry originally claimed 'turbo_kv beats fp32 KV speed'. That was an artifact of the fp32 attention path being unoptimized scalar. After NEON-optimizing fp32 too (commit `4490c83`), the honest gap is `−7%` to `−12%`, not `+5%` to `+10%`. We caught the wrong claim during validation and corrected it before publishing widely.

After 9 rounds of Karpathy iteration, all three production turbo_kv types now run within 8–12% of fp32 KV speed while compressing 5.8–9.1×:

| Type | Bytes/block | tok/s | vs FP32 | PPL | Δ vs FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 KV (NEON) | — | **14.83** | baseline | 13.56 | — |
| **`turbo_kv_4b`** ⭐ | 72 | 13.67 | **−7.8%** | 14.33 | +5.7% |
| **`turbo_kv_3b`** | 56 | 13.4 | −9.6% | 15.36 | +13.3% |
| **`turbo_kv_5b`** 🏆 | 88 | 13.13 | −11.5% | 13.65 | +0.7% |

### What changed (Round 5: the real bottleneck)

The biggest win came from `tq_transformer.c`. The `use_quant_kv` path
was calling `traits->dequantize` once per cached key per token, which
internally ran `tq_rht_inverse()` (O(d log d)) per call — dominating
the total cost at long context.

Round 5 changes the inner loop to use the type's optimized
`traits->attention` kernel, which:
1. Pre-rotates the query ONCE per layer
2. Does fused dequant + dot product per block in rotated space
3. Skips per-position inverse RHT entirely

Old slow path is preserved as a fallback for the complex cases:
QK-norm-on-stored-keys, k_highres_window, sliding-window attention.

### Karpathy loop (this release)

| Round | What changed | Llama 3.2 3B turbo_kv_4b tok/s |
|---:|---|---:|
| 0 | Baseline (per-position dequant + inline dot) | 6.9 |
| 1 | Single-pass dequant with hoisted LUT | 7.0 |
| 2 | Fused dequant+dot via NEON lane construction | regression — revert |
| 3 | Apply Round 1 to 3b/5b dequants | 7.0 |
| 4 | Pure scalar fused with 4 accumulators | 7.0 |
| 5 | **transformer uses traits->attention (no per-pos RHT inverse)** | **13.5** ✅ |
| 6 | Hoist LUT in 4bo/3bo dequants | 13.9 |

PPL changed slightly across the FP reordering (0.3–0.5% increase per
type, all within the regression test cosine ≥ 0.99/0.999 thresholds).
35/35 tests pass.

### Other changes

- New tracking issue #15 follow-up notes for per-head rotation seeds and
  Llama 3.1 8B + LongBench-E reproduction (still open)

## [0.6.2] — 2026-04-08

### Highlights

- **🆕 `turbo_kv_4bo` / `turbo_kv_3bo`** — Per-block outlier handling research types. Each block stores the K=8 channels with the largest |rotated[i]| as exact FP16 values that overwrite the codebook reconstruction at dequant time. This is a simpler local form of the per-channel outlier handling described in the Google TurboQuant paper.
- **Karpathy-loop validation**: per-channel outliers cut the PPL gap **by more than half** on Llama 3.2 3B (4b: +5.3% → 4bo: +2.2%). Effect is model-dependent — see notes below.
- **Issue #15 progress**: closes the per-channel outlier handling exploration item. 5b remains the recommended quality option; 4bo/3bo ship as experimental.

### KV quantization quality (Llama 3.2 3B, FP32 = 13.56 PPL)

| Type | Bytes/block | Compression | PPL | Δ vs FP32 | Status |
|---|---:|---:|---:|---:|---|
| `turbo_kv_3b` | 56 | 9.1× | 15.39 | +13.5% | aggressive |
| `turbo_kv_4b` ⭐ default | 72 | 7.1× | 14.28 | +5.3% | production |
| **`turbo_kv_3bo`** 🧪 | 80 | 6.4× | 14.03 | +3.5% | research |
| **`turbo_kv_5b`** 🏆 quality | 88 | 5.8× | **13.60** | **+0.34%** | production |
| **`turbo_kv_4bo`** 🧪 | 96 | 5.3× | 13.86 | +2.2% | research |

### Notes on the outlier types

Per-channel outlier handling is **data-dependent**:
- On Llama 3.2 3B (head_dim=128, heavier tails), `3bo` Pareto-improves over `4b`
- On SmolLM2 135M (smaller dimensions), `3bo` regresses past `4b` because the 3-bit base is too coarse
- `4bo` is dominated by `5b` on both models — slightly bigger and slightly worse

Until per-model auto-selection is implemented, the Pareto-optimal recommendations remain `turbo_kv_4b` (default) and `turbo_kv_5b` (quality). The outlier types are exposed for researchers and benchmarking.

## [0.6.1] — 2026-04-08

### Highlights

- **🆕 `turbo_kv_5b` — near-lossless KV** at +0.34% PPL on Llama 3.2 3B. Uses a 32-level Lloyd-Max-Gaussian codebook (Max 1960 Table I) on RHT-rotated values. 88-byte block (vs 72 for 4b). The new quality-maximizing option for users who can spare 22% more KV memory than 4b.
- **Regression tests** — three deterministic synthetic-data tests pin the attention cosine quality of `turbo_kv_4b` (≥0.99) and `turbo_kv_5b` (≥0.999), and assert 5b ≥ 4b on the same data. Future Karpathy-loop iterations cannot regress past these thresholds without failing CI.

### KV quantization quality (Llama 3.2 3B, FP32 = 13.56 PPL)

| Type | Bytes/block | Compression | PPL | Δ vs FP32 |
|---|---:|---:|---:|---:|
| `turbo_kv_3b` | 56 | 9.1× | 15.39 | +13.5% |
| `turbo_kv_4b` ⭐ default | 72 | 7.1× | 14.28 | +5.3% |
| **`turbo_kv_5b`** 🏆 | 88 | 5.8× | **13.60** | **+0.34%** |

### Tests

- `TurboKVRegression.KV_4B_AttentionCosine` — pins ≥ 0.99
- `TurboKVRegression.KV_5B_AttentionCosine` — pins ≥ 0.999
- `TurboKVRegression.KV_5B_BeatsKV_4B` — invariant: more bits ⇒ ≥ accuracy

35/35 tests pass on macOS / Linux / Windows.

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
