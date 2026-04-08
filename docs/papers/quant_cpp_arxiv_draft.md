# quant.cpp: A Single-Header C Reference Engine for KV Cache Quantization Research

> arXiv draft v0.1 — 2026-04-08
> Status: outline + section drafts; not yet ready for submission
> Target venue: arXiv cs.LG

## Abstract

We present **quant.cpp**, a single-header C reference engine for end-to-end LLM inference with extensible KV cache quantization. Through nine rounds of empirical Karpathy-loop iteration starting from a literal port of the published TurboQuant algorithm (Zandieh et al., ICLR 2026), we derive **Variant F**: a Random Hadamard Transform + scalar Lloyd-Max-Gaussian codebook with per-block max-abs scaling, structurally closest to HIGGS (Malinovskii et al., 2024) but applied to KV cache with no QJL residual or per-channel outlier handling. On Llama 3.2 3B with WikiText-style perplexity evaluation, the resulting `turbo_kv_4b` type achieves **+5.7% PPL at 7.1× compression**, running at **−7.2% throughput** vs uncompressed FP32 KV. The 5-bit variant achieves **+0.7% PPL at 5.8× compression** at −11.8% throughput. We document the full derivation history including ablations that did not work, the validation step that flipped a wrong "beats fp32" claim into the corrected "−7% vs fp32" framing, and the engineering choices that prioritize embedded portability (the entire engine compiles to a 192 KB WebAssembly binary, runs on iOS, Android, MSVC, and microcontrollers). All measurements, regression tests, and the optimization commit history are publicly auditable on GitHub.

## 1. Introduction

LLM inference is increasingly memory-bound by the KV cache. At 32K context length, an 8B model's KV cache consumes 4GB — more than the model weights themselves. Weight quantization (Q4, Q8) is well-studied; KV cache quantization is less mature in production engines.

This paper documents:

1. **An empirical derivation** of a KV cache quantizer (Variant F) starting from a literal port of Google's TurboQuant (April 2026) and arriving at a structurally simpler scheme through 9 rounds of measure-then-modify-then-revert iteration.
2. **A validation discipline** that caught and corrected a wrong performance claim before publication. The corrected version is more credible than the inflated version.
3. **An engineering case study** in keeping the implementation small enough to be readable, single-header, and dependency-free. The full engine is 72K lines of C11 with no external runtime; the single-header `quant.h` is 15.7K lines / 628 KB.
4. **A practical comparison matrix** of seven KV quantization types on three real models, with measured PPL and throughput, all publicly reproducible.

We do not claim a new algorithm. The pattern (RHT + grid quantization) was introduced by HIGGS in November 2024. The KV cache application was popularized by TurboQuant in April 2026. Our contribution is an empirically-validated simplification (drop QJL, drop outlier channels), a small portable C reference, and a transparent record of the optimization process including the failed attempts.

## 2. Background

### 2.1 KV cache memory dominates at long context

For a transformer with `L` layers, `H` attention heads, head dimension `d`, and sequence length `T`, the FP16 KV cache consumes `2 · L · H · d · T · 2` bytes per token. For Llama 3.1 8B at 32K context, this is approximately 4 GB — larger than the model weights themselves at INT8.

### 2.2 Prior art on KV cache compression

| Work | Year | Application | Method | Bit budget |
|---|---|---|---|---|
| llama.cpp Q4_0 / Q5_0 / Q8_0 KV | 2024 | KV cache | Per-block min-max linear | 4–8 bits |
| QJL [Zandieh et al.] | 2024 | KV cache | 1-bit Johnson–Lindenstrauss sign hash | 1 bit + outliers |
| PolarQuant [arXiv:2502.02617] | 2026 | KV cache | Polar coordinates `(r, θ)` quantization | 3–4 bits |
| HIGGS [Malinovskii et al.] | 2024 | **Weights** | RHT + MSE-optimal vector grids | 2–8 bits |
| TurboQuant [Zandieh et al.] | 2026 | KV cache | RHT + Lloyd-Max scalar + 1-bit QJL residual + outliers | 2.5–3.5 bits |
| **quant.cpp Variant F (this work)** | 2026 | KV cache | RHT + Lloyd-Max scalar + max-abs scaling | 3–5 bits |

### 2.3 The Random Hadamard Transform

The Random Hadamard Transform (RHT) preprocesses a vector by composing a diagonal sign matrix `D ∈ {±1}^d` with the Walsh-Hadamard transform `H_d`: `Π = (1/√d) H_d D`. This is an orthogonal transform: `Π^T Π = I`, `‖Πx‖ = ‖x‖`, and `⟨Πx, Πy⟩ = ⟨x, y⟩`.

For LLM activations and weights, the post-RHT distribution is closer to Gaussian than the original (a corollary of the Central Limit Theorem on the WHT structure). This makes scalar quantization with a fixed Gaussian-MSE-optimal codebook near-optimal, whereas the un-rotated distribution would require per-block adaptive codebook construction.

HIGGS introduced this RHT preprocessing for weight quantization in November 2024. TurboQuant adapted it to KV cache in April 2026 with a 1-bit QJL residual stage and per-channel outlier handling.

## 3. Implementation: quant.cpp

quant.cpp is a 72K-line C11 inference engine with no external dependencies beyond libc. Its design priorities, in order:

1. **Readability** — the full transformer forward pass is in one file (`src/engine/tq_transformer.c`).
2. **Embeddability** — ships as `quant.h`, a single-header library (15.7K lines, 628 KB).
3. **Portability** — runs on iOS, Android, WebAssembly (192 KB binary), MSVC Windows, and any C11 target.
4. **Quantization research** — adding a new KV quantization type requires implementing 3 functions and registering them.

### 3.1 KV quantization plugin system

Each KV quantization type registers a trait struct with three function pointers:

```c
typedef struct {
    const char* name;
    int block_size;
    size_t type_size;
    void (*quantize)(const float* src, void* dst, int n);
    void (*dequantize)(const void* src, float* dst, int n);
    void (*attention)(const float* query, const void* kv,
                      float* scores, int seq_len, int head_dim);
} tq_type_traits_t;
```

This abstraction supports both the simple round-trip path (`quantize`/`dequantize`) and the optimized fused attention path (`attention`), where the type can pre-rotate the query once and skip per-position inverse RHT.

### 3.2 Variant F derivation

Variant F was not designed top-down. It was the result of 9 rounds of Karpathy-loop iteration:

| Round | Variant | turbo_kv_4b PPL | turbo_kv_4b tok/s | Decision |
|---:|---|---:|---:|---|
| 0 | Literal port (RHT + 3-bit codebook + 1-bit QJL residual) | 16.03 | 6.9 | baseline |
| 1 | empirical std rescale | 15.87 | 7.0 | keep |
| 2 | max-abs no-clip | 15.39 | 7.0 | keep 4b only |
| 3 | 99th percentile clipping | 17.24 | — | revert |
| 4 | K·std sweep (K ∈ {1.5..4}) | 15.53 (K=2) | — | B still wins |
| 5 | uniform 8-level linear | 16.28 | — | revert |
| **6** | **Drop QJL stage, double codebook (3-bit → 4-bit)** | **14.28** | **13.5** | **shipped** |
| 7 | LUT hoist in 4bo/3bo dequant | 14.28 | 13.7 | keep |
| 8 | Gather memcpy prefetch | noise | 13.7 | revert |
| 9 | NEON fp32 baseline (validation) | — | — | adopted, fp32: 12.6→14.8 |

The single most impactful round was Round 5: dropping the QJL residual stage entirely. Ablation showed the QJL correction term contributed *byte-identical zero* to the final attention scores in our regime. Rather than debug the QJL stage, we removed it and reinvested the freed 16 bytes per block into a finer Lloyd-Max codebook (3-bit → 4-bit, 8 → 16 levels). Combined with max-abs scaling instead of theoretical √d, this single change took `turbo_kv_4b` PPL from 16.03 to 14.28 — a structural simplification, not a tuning win.

### 3.3 Validation: the fp32 baseline correction

Round 5 of the optimization Karpathy loop also accidentally produced a wrong claim. After Round 5 we measured:

- `fp32` KV: 12.6 tok/s (Llama 3.2 3B PPL eval, 28 layers, attention-heavy)
- `turbo_kv_4b`: 13.5 tok/s

We published this as "turbo_kv_4b beats fp32 KV speed at 7× compression" in the v0.6.3 release notes.

A subsequent validation pass discovered that the `fp32` attention path was using a pure scalar inner loop while the quantized path had NEON optimization. The comparison was unfair. Once we added NEON to the `fp32` path, `fp32` jumped from 12.6 → 14.83 tok/s (+18%), and the honest gap flipped:

- `fp32` KV (NEON): 14.83 tok/s baseline
- `turbo_kv_4b`: 13.57 tok/s, **−7.2%**
- `turbo_kv_5b`: 12.90 tok/s, −11.8%

We published v0.6.4 as a correction, with the wrong claim explicitly retracted in both the README and the v0.6.3 release notes. The honest framing is *"closes the speed gap from −45% to −8% with 7× compression"*, not *"beats fp32"*.

This validation step is now part of our standard process: **after any claimed performance win, re-validate the comparison baseline before publishing**. We document this lesson in the project's persistent memory and reference it in future Karpathy rounds.

## 4. Experiments

### 4.1 Setup

- **Hardware**: Apple M1 Pro, 8 threads
- **Dataset**: `bench/data/ppl_1k.txt` (1040 tokens of WikiText-style text)
- **Models**: Llama 3.2 3B Instruct, SmolLM2 135M Instruct, Gemma 4 26B-A4B-it (smoke test only)
- **Methodology**: Each measurement averaged over 3 runs. Standard deviation ~3%.
- **Quality metric**: Forward-pass perplexity via `--ppl` flag (teacher-forced)
- **Speed metric**: Tokens per second on the same PPL eval (representative of attention-heavy workloads)

### 4.2 Llama 3.2 3B Instruct results

| KV Config | Bytes/block | Compression | PPL | Δ vs FP32 | tok/s | vs FP32 speed |
|:----------|------------:|------------:|----:|----------:|------:|--------------:|
| FP32 reference (NEON) | — | 1× | 13.56 | — | 14.83 | baseline |
| `turbo_kv_5b` (quality) | 88 | 5.8× | **13.65** | **+0.7%** | 13.13 | −11.5% |
| `turbo_kv_4bo` (research) | 96 | 5.3× | 13.90 | +2.5% | 12.7 | −14% |
| `turbo_kv_4b` (default) | 72 | 7.1× | 14.33 | +5.7% | 13.67 | **−7.8%** |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 13.4 | −9.6% |
| `turbo_kv_3bo` (research) | 80 | 6.4× | 14.17 | +4.5% | 9.3 | −37% |
| `uniform_4b` (legacy) | 68 | 7.5× | 14.60 | +7.7% | 11.7 | −21% |
| llama.cpp `q4_0` KV (lit. survey) | ~70 | ~7.3× | ~14.99 | +10.6% | — | — |

The Pareto-optimal recommendations are:

- **`turbo_kv_4b`** (default): 7.1× compression, +5.7% PPL, 92% of FP32 KV speed
- **`turbo_kv_5b`** (quality): 5.8× compression, +0.7% PPL (near-lossless), 88% of FP32 KV speed

`turbo_kv_4b` strictly dominates `uniform_4b` on every relevant axis (better PPL, faster, comparable compression).

### 4.3 Validation on additional models

[TODO: Llama 3.1 8B numbers are running in background — populate when complete]

[TODO: SmolLM2 135M numbers from prior runs — already collected, populate]

### 4.4 Ablations that did not work

We document failed Karpathy rounds because the negative results are themselves informative.

- **Round 2 (NEON lane construction `{a, b, c, d}` for fused dequant + dot)**: regressed by ~7% on the smaller model. Apple Silicon's 4-element vector construction via per-lane `ins` instructions has higher latency than the L1-cache-hot two-pass pattern (separate dequant + dot product).
- **Round 3 (99th percentile clipping)**: regressed by ~12% on PPL. The clipped 1% of outliers had high quantization error that was disproportionately influential in attention scores.
- **Round 5 (uniform 8-level linear in [min, max])**: regressed by ~6%. Real key vectors after RHT remain heavier-tailed than uniform; the Gaussian-shaped Lloyd-Max codebook fits better.
- **Round 8 (gather + memcpy prefetch)**: no measurable improvement on Apple M1 Pro. The L1 prefetcher already handles the strided pattern.
- **Round 9 (strided per-position attention without gather)**: not pursued because it would require either repeated query rotation per position (slower) or a new traits ABI for pre-rotated single-block dot product (invasive without clear win).

## 5. Discussion

### 5.1 Honest attribution

The Variant F structure (RHT + Lloyd-Max scalar codebook + max-abs scaling) is not novel. The combination of RHT preprocessing and grid quantization was introduced for LLM compression by HIGGS in November 2024 (for weights). TurboQuant adapted the rotation pattern to KV cache in April 2026 with additional QJL and outlier-handling stages. Our Variant F started as a literal port of TurboQuant and converged through ablation onto a structure closer to HIGGS than to the published TurboQuant.

We credit both papers explicitly. We do not claim our shipped variant is the published TurboQuant algorithm.

### 5.2 Where is Variant F a Pareto improvement?

Variant F's strict dominance is over `uniform_4b` (a per-block min-max linear quantizer) at the same bit budget. Against `fp32` KV, Variant F is `−7%` slower at 7× compression — a meaningful compression-for-speed trade for memory-constrained deployment.

Against the published TurboQuant (which we cannot directly run for comparison), we expect Variant F to be slightly worse on quality at the same bit budget because we drop the per-channel outlier handling and the QJL residual stage. These additions bring TurboQuant near-zero PPL degradation at 3.5 bits on Llama 3.1 8B [Zandieh et al., 2026]. Variant F achieves +5.7% PPL at 4 bits / +0.7% at 5 bits on Llama 3.2 3B. The papers' numbers are not directly comparable due to model and benchmark differences; this is a future work item.

### 5.3 Embedded niche

A central design constraint of quant.cpp is single-header portability. The 192 KB WebAssembly binary, the iOS / Android / MSVC support, and the absence of any framework dependency are deliberate choices that exclude many research-grade techniques (e.g., learned codebooks, per-token routing) that would require runtime infrastructure beyond `libc + libm + pthreads`. Variant F was selected partly because it fits into 64 bytes of inline state per 128-element block with no auxiliary tables.

### 5.4 What we learned about Karpathy-loop discipline

Two lessons stand out:

1. **The most impactful round was a structural change found by ablation, not by incremental tuning.** Round 5 dropped a stage entirely after measuring it contributed zero. Rounds 1-4 each added marginal local improvements. The local optimization approach hit diminishing returns; the structural simplification was the breakthrough.

2. **Validating the comparison baseline is as important as validating the optimization itself.** Round 5 produced a wrong "beats fp32" claim because the fp32 baseline was unoptimized. The Karpathy loop's discipline of *measure → modify → measure → revert if worse* needs an additional step: *→ validate the baseline → publish only after*.

These lessons are recorded in the project's persistent memory and applied prospectively to future optimization work.

## 6. Related Work

[TODO: expand with full citations and discussion]

- HIGGS (Malinovskii et al., 2024) — RHT + MSE-optimal grids for weight quantization
- TurboQuant (Zandieh et al., 2026) — RHT + Lloyd-Max + 1-bit QJL + outliers for KV cache
- PolarQuant (2026) — Polar coordinate KV quantization
- QJL (Zandieh, 2024) — 1-bit JL sketch
- KIVI, KVQuant, ReKV, GEAR — other KV cache quantization works to survey
- llama.cpp Q4_0/Q5_0/Q8_0 KV — production baselines

## 7. Reproducibility

All measurements in this paper are reproducible from the public repository at https://github.com/quantumaikr/quant.cpp .

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download a model
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

# Reproduce the headline table
for k in fp32 turbo_kv_4b turbo_kv_5b turbo_kv_3b uniform_4b; do
    ./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf \
        --ppl bench/data/ppl_1k.txt -j 8 -k $k -v fp16
done
```

The full Karpathy-loop history is in `bench/results/turboquant_reproduction.md` with commit hashes for every round. The validation correction is in commits `4490c83` and `33b6315`. Regression tests pin attention cosine similarity above 0.99 (4-bit) and 0.999 (5-bit) — see `tests/test_turbo_kv.cpp::TurboKVRegression`.

## Acknowledgements

Tim Dettmers ([discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969)) for pointing out the HIGGS attribution. Mohamed Chorfa for the bug fix PRs (#12, #13). The ggml-org / llama.cpp community for the Discussion #20969 venue for KV quantization research.

## References

[TODO: format properly]

- Malinovskii, V., Panferov, A., Ilin, I., Guo, H., Richtárik, P., Alistarh, D. (2024). Pushing the Limits of Large Language Model Quantization via the Linearity Theorem. arXiv:2411.17525.
- Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. (2026). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. ICLR 2026. arXiv:2504.19874.
- Zandieh, A. (2024). Quantized Johnson–Lindenstrauss Transform for KV Cache Compression. AAAI 2025. arXiv:2406.03482.
- (PolarQuant). Quantizing KV Caches with Polar Transformation. AISTATS 2026. arXiv:2502.02617.
