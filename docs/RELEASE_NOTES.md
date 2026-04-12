# Release Notes

All notable changes to quant.cpp are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.13.0] — 2026-04-12

### Highlights

**Phi-3 / Phi-3.5 architecture fully supported** — the highest-value model quant.cpp was missing. Phi-3.5-mini (3.8B params, vocab 32K) is now the recommended default, delivering the best speed/quality combo:

```bash
pip install quantcpp
quantcpp                  # downloads Phi-3.5-mini Q8_0 (~3.8 GB), starts chat
```

### Added
- **Phi-3 / Phi-3.5 architecture support** — fused QKV projection, fused gate+up FFN, LongRoPE with NeoX-style rotation. Validated end-to-end on Phi-3.5-mini-instruct-Q4_K_M and Q8_0.
- **Phi-3.5-mini as default model** — replaces SmolLM2-1.7B as the recommended model. Q8_0 variant is 2x faster than Q4_K_M on Apple Silicon NEON (3.0 vs 1.5 tok/s).
- **ChatML template marker filter** — 32-byte lookahead filter in `chat_accum_callback` catches BPE-split markers (`<|im_start|>`, `<|im_end|>`, `<end_of_turn>` etc.) across token boundaries. Prevents template tokens from leaking into chat output.
- **Unsupported architecture hard-fail** — loading a model with fused QKV that quant.cpp can't handle (e.g., before Phi-3 support) now fails fast with a clear error message instead of silently producing garbage tokens.
- **quant-server-unified** — new server binary built directly on `quant.h` (single-header amalgamation). Eliminates divergence between `quant.h` and `libturboquant` split sources. CLI `quantcpp serve` now prefers this binary.
- **SmolLM2-1.7B** and **Phi-3.5-mini** added to `_MODEL_REGISTRY` with CLI aliases (`smollm2`, `phi3.5`, `phi-3.5-mini` etc.).
- **`ChatContextOverflow` exception** — Python `Model.chat()` now raises a typed exception on context overflow instead of silently returning empty output.
- **`docs/supported_models.md`** — architecture compatibility matrix, vocab-size speed guide, model selection recommendations.
- **`tools/gguf_inspect.c`** — GGUF tensor/metadata inspector for architecture debugging.

### Fixed
- **16 chat-cache bugs eliminated** (PRs #52, #53) — two audit passes found hidden bugs in KV cache prefix matching, text accumulation, server session management, WASM state handling.
- **`tq_generate_continue` overflow** — sliding-window truncation silently desynced `cached_text` from KV positions → garbage on long histories. Now returns `-2` on overflow.
- **`chat_accum_callback` realloc failure** — silently dropped tokens AND skipped user callback. Now always passes tokens through; marks accumulator tainted.
- **Server error handling** — `gen_rc == -1` produced HTTP 200 with empty content; now returns HTTP 500 with error JSON. Streaming sends `finish_reason: "error"`.
- **Server session kv_type mismatch** — reusing a session ID with different `kv_type`/`value_quant_bits` corrupted KV blocks. Now detects and rebuilds.
- **WASM `wasm_load_model`** — didn't reset `g_generating` flag → stuck busy after interrupted run.
- **`rep_penalty` in fast-path** — silently ignored in `tq_generate_chat_text`'s fast path (slow path applied it). Now consistent.
- **BOS token for Phi-3/Llama** — `<s>` added to BOS lookup chain. Phi-3 produces garbage without BOS.
- **Python CLI overflow handling** — `cmd_run` caught `ChatContextOverflow`, drops oldest turn, retries.

### Changed
- Default model: `Llama-3.2-1B` → `SmolLM2-1.7B` → **`Phi-3.5-mini` Q8_0**.
- CLI examples and README quickstart updated to use Phi-3.5-mini.
- Metal GPU dispatch disabled for fused-tensor models (CPU is faster for sub-4B).

### Performance
- **Phi-3.5-mini Q8_0**: 3.0 tok/s on Apple M3 (2x faster than Q4_K_M).
- **Chat KV cache reuse**: turn N+1 prefill is O(new tokens), not O(history). ~50% latency reduction on multi-turn chat.

---

## [v0.3.0] — 2026-04-01

### Highlights

**Real-model validation**, **adaptive compression**, and **information-theoretic foundations**. Every theoretical claim is now backed by measured data from actual model inference.

### Added

#### Real-Model Validation (Phase A)
- **Perplexity pipeline** (`--ppl <file>`): Teacher-forced PPL measurement. Gemma 4B results: 1-bit K + Q4 V PPL = 36.00 vs FP16 PPL = 35.99 — **+0.03% degradation** (effectively lossless).
- **Formal unbiasedness** (`tests/test_unbiased.cpp`): 100K random vector pairs prove all quant.cpp types have < 0.2% relative bias. The "unbiased inner product" claim is empirically verified.
- **Activation profiling** (`--profile-kv`): Per-layer pre/post-RHT distribution statistics. RHT reduces kurtosis from 10-99 to 3.9-7.9 and eliminates skewness. Honest finding: post-RHT is not perfectly Gaussian.
- **Memory bandwidth benchmark** (`--bench-memory`): tok/s vs context length across KV types.

#### Adaptive Compression (Phase B)
- **Per-layer bit recommendation** (`--recommend`): Profiles activation kurtosis, recommends 1-bit or 3-bit per layer. Gemma 270M: average 2.0 bits (vs 3.0 uniform) → 33% memory savings potential.
- **Attention entropy analysis** (`--attn-entropy`): Per-head Shannon entropy identifies sharp vs diffuse attention patterns.
- **V highres window** (`-V N`): Recent N tokens stored as FP16 alongside Q4/Q2 V. Test showed Q4 V already near-lossless (PPL +0.03%), so hybrid adds no measurable benefit.
- **Online codebook calibration** (`--calibrate`): Lloyd-Max iteration on real activation data. **MSE improved 49.7%** over default N(0,1) codebook — proves model-specific calibration matters.

#### Engine (Phase C)
- **Fused Q4 domain attention**: Weighted sum computed directly from packed nibbles without dequantize buffer. NEON `vfmaq_f32` path. Reduces memory traffic.
- **Prefill benchmark** (`--bench-prefill`): Measures KV quantization overhead during prompt processing.
- **CoW benchmark** (`bench/cow_bench.sh`): Analytical memory savings for shared-prefix serving.
- **Auto compression profile** (`bench/auto_profile.sh`): Full pipeline: profile → recommend → calibrate → JSON output.

#### Theory (Phase D)
- **Rate-distortion bounds** (`tests/test_rate_distortion.cpp`): Computes info-theoretic minimum MSE at each bit-width. Q4 uniform: 2.41x gap. Lloyd-Max: < 0.15 bits wasted.
- **Cumulative error analysis** (`tests/test_cumulative_error.cpp`): 16-layer simulation shows errors grow sub-linearly. Cosine similarity after 16 layers: 0.998 (Q4), 0.951 (Q2).

### Measured Results

| Metric | Value | Source |
|--------|-------|--------|
| Gemma 4B PPL (uniform_4b) | 35.99 | `--ppl` |
| Gemma 4B PPL (1b K + Q4 V) | 36.00 (+0.03%) | `--ppl` |
| Gemma 4B PPL (1b K + Q2 V) | 42.23 (+17.3%) | `--ppl` |
| Unbiasedness (all types) | < 0.2% rel_bias | `test_unbiased` |
| Post-RHT kurtosis range | 3.9 – 7.9 | `--profile-kv` |
| Adaptive bit average | 2.0 bits (33% saving) | `--recommend` |
| Calibrated codebook MSE improvement | 49.7% | `--calibrate` |
| 16-layer cumulative cosine (Q4) | 0.998 | `test_cumulative_error` |
| Rate-distortion gap (Q4 uniform) | 2.41x | `test_rate_distortion` |

---

## [v0.2.0] — 2026-04-01

### Highlights

**V cache quantization** and **expert-grade validation** — total K+V compression reaches 4.9x (Q4) to 7.1x (Q2), with every claim backed by measured data.

### Added

#### V Cache Quantization
- **Q4 value quantization** (`-v q4`): 4-bit per-block scale + packed nibbles. V compression 3.8x.
- **Q2 value quantization** (`-v q2`): 2-bit Lloyd-Max codebook. V compression 7.6x.
- **FP16 value auto-enable**: Values stored as FP16 when KV quantization is active (was FP32).
- Combined 1-bit K + Q4 V: **27.62 KB/token, 4.9x total K+V** (was 136 KB FP16).
- Combined 1-bit K + Q2 V: **19.12 KB/token, 7.1x total K+V**.
- CLI flag `-v q4|q2|fp16` for value quantization control.
- Memory reporting (`-M`) shows K and V breakdown separately.

#### Validation Suite
- **NEON/scalar consistency** (`tests/test_neon_scalar.cpp`): 14 tests verify every NEON path against pure C reference — Q4 dequant, Q2 dequant, RHT butterfly, RoPE, matmul, RMSNorm, Hamming attention.
- **Attention distribution** (`tests/test_attention_distribution.cpp`): 8 tests measure cosine similarity (0.996/0.918/0.634), Spearman rank correlation, top-k overlap. Proves compression is non-trivial (random K = 0.089).
- **Codebook theory** (`tests/test_codebook_theory.cpp`): 5 tests verify Lloyd-Max centroids match N(0,1) literature values within 0.001, MSE within 1.18x of information-theoretic optimal.
- **Edge cases** (`tests/test_edge_cases.cpp`): 29 tests — n=1 (single token), dim=0, NaN input, Inf input, all-same values, all-zero, n=10000 large sequence.
- **Numerical stability**: 4 tests for overflow-safe norm computation and NaN/Inf input guards.

#### Benchmark Scripts
- `bench/ablation_test.sh`: Divergence analysis at 50-300 tokens across KV types.
- `bench/long_quality_test.sh`: Coherence at 200/500/1000 tokens.
- `bench/sampling_test.sh`: Temperature sampling (T=0.3, T=0.7) comparison.
- `bench/quant_time_bench.sh`: Quantization timing wrapper.
- `bench/bench_kv_overhead.cpp`: Microbenchmark — uniform 148 ns, 1b 659 ns, 3b 11066 ns per vector.
- `bench/attention_dist_test.sh`: Attention distribution analysis wrapper.
- `scripts/sanitize.sh`: ASan + UBSan build and full test run.

### Fixed

- **Q4 dequant NEON nibble interleaving bug**: Lo/hi nibbles were written contiguously instead of interleaved, causing MSE 0.525 (300x worse than correct). Fixed with `vzip_u8` interleave.
- **QJL sign bias**: `proj >= 0.0f` → `proj > 0.0f` across 11 occurrences (CPU, CUDA, Metal). Eliminates asymmetric bias at zero projection boundary.
- **Norm overflow**: QJL norm computation now uses max-abs rescaling to prevent float overflow on large vectors.
- **NaN/Inf input guard**: Quantization functions zero-fill output block on NaN/Inf input instead of producing undefined output.

### Changed

- **Thread safety**: Global Q8 workspace (`g_q8_buf`) and sampler probability index (`g_probindex`) protected by mutex against concurrent realloc races.
- **RHT NEON vectorized**: Walsh-Hadamard butterfly uses `float32x4_t` for stages with len >= 4.
- **Q4 dequant NEON restored**: Properly vectorized with `vzip_u8` after bug fix (was scalar fallback).
- Test suite count: 23 → 26. Edge case count: 16 → 29.

### Measured Results

| Metric | Value | Source |
|--------|-------|--------|
| Total K+V compression (1b K + Q4 V) | 4.9x | `quant -M` |
| Total K+V compression (1b K + Q2 V) | 7.1x | `quant -M` |
| 32K context savings (Q4 V) | 3.4 GB | calculated |
| Attention cosine (uniform_4b) | 0.996 | `test_attention_distribution` |
| Attention cosine (turbo_kv_3b) | 0.918 | `test_attention_distribution` |
| Attention cosine (turbo_kv_1b) | 0.634 (= 2/pi) | `test_attention_distribution` |
| Random K cosine | 0.089 | `test_attention_distribution` |
| Lloyd-Max MSE vs theory | < 1.18x | `test_codebook_theory` |
| RHT overhead | 147 ns/vec | `bench_kv_overhead` |
| 1-bit attention | 1.2 ns/key | `bench_kv_overhead` |
| ASan + UBSan | 26/26 clean | `scripts/sanitize.sh` |

---

## [v0.1.0] — 2026-03-31

### Highlights

**Initial release** — pure C inference engine with quant.cpp KV cache compression. 1-bit keys, 10.7x key compression, byte-identical greedy output at 100 tokens.

### Added

#### Core Engine
- Complete transformer inference engine in pure C11 (10,000+ lines).
- Multi-architecture support: Gemma 3 (sliding window, GeGLU, dual RoPE) + Qwen3.5 (DeltaNet hybrid).
- Multi-shard safetensors loading (Gemma 4B = 2 shards, 883 tensors).
- Dual tokenizer: GPT2 byte-level BPE + SentencePiece auto-detect.
- TQM binary format: pre-quantized mmap, instant loading.

#### KV Cache Quantization (11 types)
- **quant.cpp KV 1-bit**: Sign-only after RHT. XOR + popcount attention (NEON `vcntq_u8`).
- **quant.cpp KV 3-bit**: 2-bit Lloyd-Max codebook + 1-bit QJL residual.
- **quant.cpp KV 4-bit**: 3-bit codebook + 1-bit QJL.
- **Uniform 4-bit / 2-bit**: Standard min-max quantization.
- **PolarQuant**: Polar coordinate (theta + radius) quantization.
- **QJL**: Quantized Johnson-Lindenstrauss sign hash.
- **Mixed / quant.cpp base**: Combined polar + QJL.

#### Weight Quantization
- Q4 weight quantization (4-bit per-block).
- Q2 weight quantization (2-bit Lloyd-Max codebook, Q2xQ8 integer matmul).
- BF16 weight support.

#### Performance
- NEON vectorized: 2-row matmul batching, fused dot products, Hamming distance.
- Thread pool with configurable thread count.
- Apple Silicon optimized.

#### Quality Verification
- 30/30 byte-identical greedy matches (K-only, 100 tokens, 10 diverse prompts).
- 23 test suites (Google Test).
- Qwen3.5: 0.999 cosine similarity vs PyTorch reference.
- Gemma 270M: per-layer exact match.

### Models Verified

| Model | Params | Speed (Q4, 6T) |
|-------|--------|----------------|
| Gemma 3 4B | 4B | 20.2 tok/s |
| Qwen3.5-0.8B | 752M | 80.1 tok/s |
| Gemma 3 270M | 270M | 176 tok/s |

---

## Release Process

### Version Scheme

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: Breaking API changes
MINOR: New features, backward-compatible
PATCH: Bug fixes, performance improvements
```

### Checklist for New Releases

1. Update version in `CMakeLists.txt` (`project(turboquant VERSION x.y.z)`)
2. Add release section to this file (newest first)
3. Update badge version in `README.md` and `README.ko.md`
4. Run full validation:
   ```bash
   cmake --build build -j$(nproc) && ctest --test-dir build
   bash scripts/sanitize.sh
   ./build/quant gemma3-4b.tqm -p "The capital of France is" -j 6 -n 20 -T 0.0 -k turbo_kv_1b -v q4
   ```
5. Tag: `git tag -a v0.x.0 -m "Release v0.x.0"`
6. Push: `git push origin v0.x.0`
7. Create GitHub release with this section's content

### What Goes in Release Notes

- **Added**: New features, new tests, new benchmarks
- **Fixed**: Bug fixes (with root cause and impact)
- **Changed**: Behavior changes, performance improvements
- **Measured Results**: Table of key metrics with source (test name or script)
- **Breaking**: API changes that require user code modification
