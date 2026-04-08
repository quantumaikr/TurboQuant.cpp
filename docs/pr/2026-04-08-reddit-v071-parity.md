# r/LocalLLaMA — quant.cpp v0.7.1 — KV cache compression at fp32 KV speed (single-header C, 11 Karpathy rounds)

## Title (≤ 300 chars)

quant.cpp v0.7.1: I spent 4 sessions optimizing a single-header C KV cache quantizer. Round 10 finally hit fp32 KV speed parity at 7.1× compression on Llama 3.2 3B. Honest write-up with 4 corrections we caught before publishing.

## Body

**TL;DR**: Single-header (628 KB) C reference engine for KV cache quantization. After 11 Karpathy-loop rounds, `turbo_kv_4b` matches uncompressed FP32 KV speed (−1.4% within noise) at **7.1× memory compression** with **+3.8% PPL** trade-off on Llama 3.2 3B. Built CPU-only, runs on iOS/Android/WASM/MSVC/microcontrollers. Apache 2.0. https://github.com/quantumaikr/quant.cpp

---

### What this is

quant.cpp is a small C inference engine I've been working on, focused on **KV cache quantization research**. It started as a literal port of the [TurboQuant paper (Zandieh et al., ICLR 2026)](https://arxiv.org/abs/2504.19874) and converged through 11 rounds of measurement-driven iteration into something simpler that I wanted to share.

The differentiator is **single-header portability**. The whole engine is one 628 KB `quant.h` you can drop into any C/C++ project (no Cargo, no Python, no PyTorch, no framework). Build with `cc app.c -lm -lpthread` and you have a working LLM with 7× compressed KV cache. It runs on iOS, Android, WASM (192 KB binary), MSVC, microcontrollers.

### The headline result (Llama 3.2 3B Instruct, CPU-only build, 3-run average)

| KV type | Bytes/block | Compression | PPL | Δ vs FP32 | tok/s | vs FP32 speed |
|---|---:|---:|---:|---:|---:|---:|
| FP32 KV | — | 1× | 13.56 | — | 18.43 | baseline |
| **`turbo_kv_4b`** ⭐ default | **72** | **7.1×** | 14.08 | **+3.8%** | **18.17** | **−1.4%** ✅ |
| `turbo_kv_5b` 🏆 quality | 88 | 5.8× | 13.65 | **+0.7%** | 16.80 | −8.8% |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 16.57 | −10.1% |
| `uniform_4b` (legacy) | 68 | 7.5× | 14.60 | +7.7% | 13.27 | −26.8% |

`turbo_kv_4b` is now Pareto-dominant over `uniform_4b` on every axis (better PPL, faster, comparable compression). And it's at **fp32 KV speed parity** while compressing 7.1×.

### The journey (11 rounds, 4 sessions, 4 honest corrections)

This isn't a "tada, I built a thing" post. It's a record of measurement discipline.

**Round 0** — Literal TurboQuant port: PPL 16.03, way slower than `uniform_4b`. Embarrassing.

**Round 6 (Variant F)** — Karpathy ablation revealed the QJL residual stage contributed *byte-identical zero* to attention scores. Dropped it, reinvested 16 bytes per block in a finer Lloyd-Max codebook (3-bit → 4-bit, 8 → 16 levels). PPL 16.03 → 14.28. Structural simplification, not tuning.

**Rounds 7–9** — Local fusions, NEON unroll, LUT hoisting, prefetch. Each gave at most +5%. Stuck at −7% vs fp32.

**Round 10 — the breakthrough**. After three sessions of guessing, I finally ran the existing `--profile` flag. The data was unambiguous: matmul was identical between fp32 and quant (38.6 vs 38.9 ms, both share the same NEON tbl matmul kernel). The entire 8% speed gap was in the attention dot-product loop. The fp32 path was 4-way NEON SIMD; mine was scalar. ~2× more instructions per element. **Compute-bound, not memory-bound** — surprising for a 16-entry LUT.

The fix: Apple Silicon's `vqtbl1q_s8`, a single instruction that does 16 byte-table lookups across 16 lanes. Quantize the 16 Lloyd-Max-Gaussian centroids to int8 once at startup (~1% precision loss, well below the regression test cosine ≥ 0.99 threshold), store them in a 16-byte register, and the inner loop becomes:

```c
uint8x16_t bytes = vld1q_u8(mi);                    // 16B = 32 nibbles
uint8x16_t low_nib  = vandq_u8(bytes, vdupq_n_u8(0x0F));
uint8x16_t high_nib = vshrq_n_u8(bytes, 4);
int8x16_t low_vals  = vqtbl1q_s8(cb_vec, low_nib);  // 1 instr, 16 gathers
int8x16_t high_vals = vqtbl1q_s8(cb_vec, high_nib);
// ... interleave + int8→fp32 + per-block scale + vfmaq_f32
```

32 elements per inner-loop iteration (vs 8 in the previous scalar version). Result: **fp32 parity**, +4.5% on a single representative run, +0.8% on 3-run average. PPL also slightly improved (the int8 codebook discretization happens to align favorably).

**Round 11 (v0.7.1)** applied the same pattern to 5b/3b. The lookup side scales (1 instruction per 16 lanes for any small codebook) but the **bit-unpack side** is the new bottleneck: 5-bit and 3-bit indices straddle byte boundaries irregularly, so the unpack of 16 indices needs scalar shifts. 5b improved from −14.5% to −8.8% (+9% speed jump), 3b from −13% to −10%. Not full parity, but significant.

### The honest correction record (4 events)

I started this with an inflated "lossless 7×" claim and walked it back four times before publishing widely. Each correction taught a lesson now in persistent memory:

1. **v0.6.0** "lossless 7× compression" → measured "+6.3% PPL on Llama 3.2 3B"
2. **v0.6.4** "turbo_kv beats fp32 KV speed" → discovered the fp32 attention path was unoptimized scalar; once both had NEON, the honest gap was −7%
3. **v0.6.5** "with Metal" → discovered the existing Metal backend is currently *net negative* (13–40% slower) on every model size from SmolLM 135M to Gemma 4 26B due to per-matmul dispatch overhead. CMake default is OFF, but our internal benchmarks had been wrong by 14–22% for 5 releases. [Filed issue #16](https://github.com/quantumaikr/quant.cpp/issues/16).
4. **v0.6.5 post**: [@TimDettmers](https://github.com/TimDettmers) (HIGGS / QLoRA / bitsandbytes) commented in a [llama.cpp discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969) — not directly addressed to us, but the substance applied — that the RHT + scalar grid pattern we were calling "TurboQuant" was actually originally HIGGS (Malinovskii et al., Nov 2024). We updated all docs to credit HIGGS within 24 hours and reframed "Tim gave us feedback" to "Tim's general comment we observed" once a user pointed out we'd overstated the relationship.

If you're skeptical of any number above, **all measurements are reproducible** with `cmake -B build && cmake --build build && ./build/quant model.gguf --ppl bench/data/ppl_1k.txt -k turbo_kv_4b`.

### Honest framing (what this isn't)

- **Not a TurboQuant implementation.** Through ablation we dropped both the QJL residual and the per-channel outlier handling that the published paper uses. What we ship is structurally closer to HIGGS (RHT + scalar grid quantization) than to TurboQuant. Both are credited in our docs.
- **Not the fastest GPU inference.** llama.cpp owns that with full Metal/CUDA tensor graphs. We're CPU-only and proud of it.
- **Not the most feature-complete.** 7 architectures verified, not 100+. Single-header constraint excludes many features.
- **Not validated on Llama 3.1 8B yet** (the paper baseline). We tried — Q8_0 hit swap on 16 GB RAM, Q4_K_M was prohibitively slow. Tracked as TODO.
- **Not at parity for 5b/3b yet.** Round 11 closed the gap significantly but they're at −9% / −10%. Future work.

### Cross-size validation (3 Llama-family models, all CPU-only)

| Model | turbo_kv_4b PPL Δ | turbo_kv_5b PPL Δ |
|---|---|---|
| SmolLM2 135M | +5.8% | +1.7% |
| Llama 3.2 1B | +7.3% | **+0.7%** |
| Llama 3.2 3B | +5.7% | **+0.7%** |

`turbo_kv_5b` is consistently near-lossless across model sizes (~1% PPL Δ).

### Try it

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release   # default: TQ_BUILD_METAL=OFF
cmake --build build -j

# Download a small model
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "Hello!" -j 8
```

`turbo_kv_4b` is the default. Use `-k turbo_kv_5b` for near-lossless quality, `-k turbo_kv_3b` for max compression.

### Where the value is

Honestly, the 7.1× compression at fp32 parity is the headline number. But after 4 sessions, what I think is more valuable is the **measurement transparency**. Every claim links to a reproduction script. Every release notes corrections from the previous release. The 11-round Karpathy history with commit hashes is in [`bench/results/turboquant_reproduction.md`](https://github.com/quantumaikr/quant.cpp/blob/main/bench/results/turboquant_reproduction.md). If a future paper wants to cite a "single-header C reference implementation of HIGGS-style KV quantization", this is it.

### Roadmap (next sessions)

- v0.7.2: 5b 1-byte-per-index variant for full parity (trade compression for speed)
- v0.8.0: AVX2 + WASM SIMD ports of the NEON tbl pattern
- v0.9.0: `vusdotq` exploration to potentially exceed fp32 (ARMv8.6+)
- v1.0.0: arXiv submission + spec compliance test suite + llama.cpp PR

### Links

- Repo: https://github.com/quantumaikr/quant.cpp
- v0.7.1 release notes: https://github.com/quantumaikr/quant.cpp/releases/tag/v0.7.1
- Round 10 commit: https://github.com/quantumaikr/quant.cpp/commit/2537a12
- llama.cpp discussion thread we participate in: https://github.com/ggml-org/llama.cpp/discussions/20969
- Reproduction history: https://github.com/quantumaikr/quant.cpp/blob/main/bench/results/turboquant_reproduction.md

Critical feedback welcome. Especially:
- Cross-implementation comparisons (MLX, Rust forks, llama.cpp turboquant forks) on the same hardware
- Anyone with Llama 3.1 8B running quant.cpp on a 32+ GB box
- AVX2 / SIMD128 implementations of the same pattern
- Suggestions for the 5b/3b unpack bottleneck (SIMD bit-extraction tricks?)
