# TurboQuant Paper Deep Analysis & Implementation Gap Assessment

**Paper**: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
**Authors**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research / Google DeepMind)
**Published**: arXiv 2504.19874, April 2025 (ICLR 2026 accepted)

---

## 1. Paper Core Algorithm

TurboQuant is a **two-stage** vector quantization algorithm:

### Stage 1: TurboQuant_mse (MSE-optimal quantizer)

**Algorithm 1 — Quantize:**
1. Generate random rotation matrix **Pi** (orthogonal, d x d)
2. Pre-compute **codebook** centroids c_1...c_{2^b} that minimize MSE for Beta distribution
3. Rotate input: **y** = **Pi** . **x**
4. For each coordinate j: find nearest centroid idx_j = argmin_k |y_j - c_k|
5. Output: idx (array of b-bit integers per coordinate)

**Algorithm 1 — DeQuantize:**
1. Replace each idx_j with its centroid: y_tilde_j = c_{idx_j}
2. Rotate back: **x_tilde** = **Pi**^T . **y_tilde**

**Key insight**: After random rotation, each coordinate of a unit-norm vector follows a Beta distribution that converges to N(0, 1/d) in high dimensions. This allows **independent scalar quantization per coordinate** with near-optimal MSE.

### Stage 2: TurboQuant_prod (Inner-product optimal quantizer)

**Algorithm 2 — Quantize:**
1. Apply TurboQuant_mse with bit-width **b-1** (one bit less)
2. Compute residual: **r** = **x** - DeQuant_mse(Quant_mse(**x**))
3. Apply QJL (1-bit sign hash) on residual: qjl = sign(**S** . **r**)
4. Store: (idx, qjl, ||**r**||_2)

**Algorithm 2 — DeQuantize:**
1. x_tilde_mse = DeQuant_mse(idx)
2. x_tilde_qjl = sqrt(pi/2) / d * gamma * **S**^T . qjl
3. Output: x_tilde_mse + x_tilde_qjl

**Key insight**: MSE-optimal quantizers are **biased** for inner product estimation. The QJL residual correction is **unbiased**, combining both gives optimal inner product distortion.

---

## 2. Theoretical Guarantees

### MSE Distortion (Theorem 1)
- D_mse <= (sqrt(3)*pi/2) * (1/4^b) for any bit-width b >= 0
- For b=1,2,3,4: D_mse ~ 0.36, 0.117, 0.03, 0.009

### Inner Product Distortion (Theorem 2)
- **Unbiased**: E[<y, x_tilde>] = <y, x> (exact)
- D_prod <= (sqrt(3)*pi^2 * ||y||^2) / d * (1/4^b)
- For b=1,2,3,4: D_prod ~ 1.57/d, 0.56/d, 0.18/d, 0.047/d

### Lower Bound (Theorem 3)
- D_mse >= 1/4^b (information-theoretic)
- TurboQuant is within factor **sqrt(3)*pi/2 ~ 2.7** of optimal

### KV Cache Results
- **3.5 bits/channel**: absolute quality neutrality (no degradation)
- **2.5 bits/channel**: marginal quality degradation
- **4x compression** with perfect Needle-in-a-Haystack recall (score 0.997 vs 0.997 full precision)
- Outperforms KIVI, SnapKV, PyramidKV on LongBench-E

---

## 3. Paper's Outlier Treatment Strategy

The paper uses a critical strategy not obvious from the abstract:

> "Our strategy of splitting channels into **outlier and non-outlier sets**, and applying two independent instances of TurboQuant to each, allocating higher bit precision to outliers."

**2.5-bit setup**: 32 outlier channels at 3 bits + 96 regular channels at 2 bits = (32*3 + 96*2)/128 = **2.5 effective bits**
**3.5-bit setup**: Different ratio of outliers vs regular for 3.5 effective bits

This is a **mixed-precision** approach where outlier channels get more bits.

---

## 4. Gap Analysis: Paper vs Current Implementation

### 4.1 Random Rotation (CRITICAL GAP)

**Paper**: Uses random orthogonal matrix **Pi** (d x d) to rotate input vectors before scalar quantization. This is the **foundation** of the algorithm — it converts any worst-case input into a near-Gaussian distribution that enables optimal scalar quantization.

**Our implementation**: `tq_rht.c` implements Walsh-Hadamard Transform (WHT) with random sign flip. This is a **fast approximation** of random rotation (O(d log d) vs O(d^2)), which is acceptable for practical use. However:

| Aspect | Paper | Our Code | Gap |
|--------|-------|----------|-----|
| Rotation type | Full random orthogonal **Pi** | Walsh-Hadamard + random signs | Acceptable (WHT is standard practice) |
| Applied to KV cache? | Yes, before quantization | **tq_rht.c exists but NOT wired into KV quantization pipeline** | **CRITICAL: RHT is implemented but unused in the engine** |
| Pre-compute | Generate once, reuse | Seed-based deterministic | OK |

**Action**: Wire `tq_rht_transform()` into the KV cache quantization path (before `tq_uniform_4b_quantize` or `tq_polar_quantize`).

### 4.2 Codebook Design (SIGNIFICANT GAP)

**Paper**: Solves the **continuous k-means optimization** (Eq. 4) for the Beta distribution f_X(x) to find optimal centroids. For b=1: centroids = {+/- sqrt(2/(pi*d))}, for b=2: centroids = {+/- 0.453/sqrt(d), +/- 1.51/sqrt(d)}.

**Our implementation**: Uses **uniform min-max quantization** (`tq_uniform.c`): scale = (max-min)/levels, q = round(x/scale). This is the simplest possible quantizer.

| Aspect | Paper | Our Code | Gap |
|--------|-------|----------|-----|
| Quantizer type | **Optimal Lloyd-Max** for Beta/Gaussian distribution | Uniform min-max | **SIGNIFICANT: ~20-30% worse MSE** |
| Centroids | Pre-computed optimal for each bit-width | Uniformly spaced | Missing |
| Distribution-aware | Yes (tuned for post-rotation Gaussian) | No (data-agnostic) | Key gap |

**Action**: Implement optimal codebook (Lloyd-Max centroids for Gaussian) as a lookup table. For high-d, Gaussian centroids are well-known:
- b=1: {-0.7979, +0.7979} (scaled by 1/sqrt(d))
- b=2: {-1.510, -0.4528, +0.4528, +1.510} (scaled by 1/sqrt(d))
- b=3: 8 centroids from standard tables
- b=4: 16 centroids from standard tables

### 4.3 Two-Stage Quantization (SIGNIFICANT GAP)

**Paper**: Stage 1 (MSE quantizer, b-1 bits) + Stage 2 (QJL on residual, 1 bit) = b total bits. This produces **unbiased** inner product estimates.

**Our implementation**: `tq_turbo.c` does implement the two-stage pattern:
```c
tq_polar_quantize_ref(src, &block->polar, dim);  // Stage 1
// compute residual
tq_qjl_quantize_ref(residual, &block->residual, dim);  // Stage 2
```

But there are gaps:

| Aspect | Paper | Our Code | Gap |
|--------|-------|----------|-----|
| Stage 1 quantizer | Optimal Lloyd-Max after rotation | PolarQuant (atan2-based, NOT rotation-based) | **WRONG algorithm** |
| Residual computation | r = x - DeQuant_mse(Quant_mse(x)) | r = src - dequantized_polar | Correct structure |
| QJL implementation | sign(**S** . **r**) with Gaussian **S** | sign(random_projection) with Rademacher | Acceptable (Rademacher is simpler) |
| Norm storage | ||r||_2 stored explicitly | Stored in block | OK |
| DeQuant formula | x_mse + sqrt(pi/2)/d * gamma * **S**^T . qjl | Different reconstruction | Needs verification |

**Critical issue**: Our "PolarQuant" uses atan2-based polar coordinates (angle + radius), which is a **completely different algorithm** from the paper's rotation + scalar quantization. The paper's "PolarQuant" reference [28] is the same group's earlier work, but the TurboQuant paper supersedes it with the rotation-based approach.

### 4.4 QJL Implementation

**Paper**: Q_qjl(x) = sign(**S** . x), where **S** has i.i.d. N(0,1) entries. DeQuant: sqrt(pi/2)/d * **S**^T . z.

**Our implementation**: Uses Rademacher (+1/-1) random entries instead of Gaussian. This is a valid simplification (both satisfy JL property), but the dequantization formula may differ.

| Aspect | Paper | Our Code | Gap |
|--------|-------|----------|-----|
| Random matrix | Gaussian N(0,1) | Rademacher (+1/-1) | Acceptable |
| Quantize | sign(**S** . x) | sign(random_projection . x) | OK |
| DeQuant scale | sqrt(pi/2) / d | Needs verification | Check |
| Bias correction | Provably unbiased | Unverified | Test needed |

### 4.5 KV Cache Integration (CRITICAL GAP)

**Paper**: Applied to KV cache quantization in LLM inference. Specifically:
- Quantize K (keys) and V (values) separately
- Apply outlier detection: split channels into outlier/non-outlier
- Different bit allocation per group
- Applied **online** during generation (not offline)
- Tested on Llama-3.1-8B and Ministral-7B at 4K-104K context

**Our implementation**: The KV cache quantization (`src/cache/`) uses `tq_uniform_4b` (simple min-max Q4) — **not the TurboQuant algorithm at all**. The sophisticated quantization types (polar, qjl, turbo) exist in `src/core/` but are **not connected to the inference engine's KV cache**.

| Aspect | Paper | Our Code | Gap |
|--------|-------|----------|-----|
| KV quantization method | TurboQuant (rotation + Lloyd-Max + QJL) | Uniform min-max Q4 | **CRITICAL: Not using TurboQuant for KV** |
| Outlier channels | Mixed-precision (3-bit outliers + 2-bit regular) | `tq_mixed.c` exists but not in engine | Not wired |
| K/V asymmetry | Separate treatment | Config flag exists | Partial |
| Online quantization | During generation | During generation | OK |

### 4.6 Attention Computation

**Paper**: For inner-product TurboQuant, attention scores are computed as:
```
<y, Q^-1(Q(x))> = <y, x_mse> + ||r|| * <y, Q_qjl^-1(Q_qjl(r))>
```

**Our implementation**: Integer Q4×Q8 attention using vdotq_s32 — optimized for uniform quantization, not for the two-stage TurboQuant scheme.

---

## 5. Implementation Priority: What to Fix

### Priority 1: Wire RHT into KV Cache (High impact, Low effort)

The Random Hadamard Transform is already implemented (`tq_rht.c`) but not used in the KV path. Adding it before quantization would improve quality significantly by making the input distribution more uniform.

```
Before: KV_fp16 → uniform_4b_quantize → stored
After:  KV_fp16 → RHT_transform → optimal_quantize → stored
        Attention: dequant → RHT_inverse → attention_score
```

### Priority 2: Optimal Codebook (High impact, Medium effort)

Replace uniform quantization with Lloyd-Max optimal centroids for the post-rotation Gaussian distribution. This is a lookup table — the centroids are precomputed constants.

For 4-bit (16 levels) Gaussian quantizer, the optimal centroids and boundaries are well-known from quantization theory. This alone can reduce MSE by **20-30%** vs uniform.

### Priority 3: True TurboQuant Two-Stage (High impact, High effort)

Implement the actual paper algorithm:
1. Apply RHT
2. Scalar quantize with optimal codebook (b-1 bits)
3. Compute residual
4. Apply QJL on residual (1 bit)
5. Store: indices + qjl_signs + residual_norm

This would make TurboQuant.cpp a **faithful implementation** of the paper, not just named after it.

### Priority 4: Mixed-Precision Outlier Channels (Medium impact, Medium effort)

Split KV channels into outlier (high-variance) and non-outlier groups. Allocate 3 bits to outliers, 2 bits to others. This is what the paper does for their 2.5-bit configuration.

---

## 6. Quantitative Impact Estimates

| Improvement | MSE Reduction | Inner Product Error | Effort |
|-------------|---------------|---------------------|--------|
| RHT pre-rotation | ~15-25% | ~15-25% | 2-3 hours |
| Optimal codebook | ~20-30% | ~20-30% | 4-6 hours |
| Two-stage (MSE + QJL) | ~40-50% | **unbiased** (vs biased) | 8-12 hours |
| Outlier mixed-precision | ~10-20% | ~10-20% | 4-6 hours |
| **Combined** | **~60-70%** | **near-optimal** | 20-30 hours |

Current uniform Q4 achieves ~3.8x compression.
Paper's TurboQuant at 3.5 bits achieves ~4.5x compression with **zero quality degradation**.
At 2.5 bits: ~6.4x compression with **marginal** quality degradation.

---

## 7. Paper's Key Numbers for Reference

### LongBench-E (Table 1, Llama-3.1-8B-Instruct)

| Method | KV Size (bits) | Average Score |
|--------|---------------|---------------|
| Full Cache | 16 | 50.06 |
| KIVI | 3 | 48.50 |
| KIVI | 5 | 50.16 |
| PolarQuant | 3.9 | 49.78 |
| **TurboQuant** | **2.5** | **49.44** |
| **TurboQuant** | **3.5** | **50.06** |

At 3.5 bits, TurboQuant matches full precision (50.06 = 50.06).
At 2.5 bits, TurboQuant still outperforms KIVI at 3 bits.

### Needle-in-a-Haystack (Figure 4)

| Method | Score |
|--------|-------|
| Full Precision | 0.997 |
| **TurboQuant** | **0.997** |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

TurboQuant achieves **identical** performance to full precision at 4x compression.

### Quantization Speed (Table 2)

| Method | d=200 | d=1536 | d=3072 |
|--------|-------|--------|--------|
| Product Quantization | 37.04s | 239.75s | 494.42s |
| RabitQ | 597.25s | 2267.59s | 3957.19s |
| **TurboQuant** | **0.0007s** | **0.0013s** | **0.0021s** |

TurboQuant is **100,000x faster** than alternatives — crucial for online KV cache quantization.

---

## 8. Recommended Implementation Roadmap

### Phase 1: Foundation (Days 1-2)
- [ ] Implement Gaussian Lloyd-Max codebook as static lookup tables (b=1,2,3,4)
- [ ] Wire RHT into KV cache quantization path
- [ ] Add `TQ_TYPE_TURBOQUANT_MSE` that uses rotation + optimal scalar quantization
- [ ] Benchmark MSE improvement vs current uniform

### Phase 2: Two-Stage (Days 3-4)
- [ ] Implement residual computation after MSE quantization
- [ ] Apply QJL on residual with correct dequantization scale (sqrt(pi/2)/d)
- [ ] Add `TQ_TYPE_TURBOQUANT_PROD` for unbiased inner product
- [ ] Verify unbiasedness with statistical tests

### Phase 3: Mixed-Precision (Days 5-6)
- [ ] Implement outlier channel detection (top-K variance channels)
- [ ] Allocate 3 bits to outliers, 2 bits to regular (2.5-bit config)
- [ ] Allocate 4 bits to outliers, 3 bits to regular (3.5-bit config)
- [ ] Benchmark on LongBench-E equivalent tasks

### Phase 4: Integration (Days 7-8)
- [ ] Replace `uniform_4b` as default KV cache type with `turboquant_3.5b`
- [ ] Update benchmarks with true TurboQuant numbers
- [ ] Compare against paper's reported results
- [ ] Update README with "faithful paper implementation" claim

---

## 9. Conclusion

**Current state**: TurboQuant.cpp is named after the paper but uses **uniform min-max quantization** for KV cache, not the actual TurboQuant algorithm. The core algorithms (polar, qjl, turbo) exist in `src/core/` but are **not connected to the inference engine**.

**Impact of fixing**: Implementing the true TurboQuant algorithm would:
1. Reduce KV cache to **2.5-3.5 bits** (vs current 4 bits) — **30-55% more compression**
2. Achieve **zero quality degradation** at 3.5 bits (vs current measurable degradation at 4 bits)
3. Make TurboQuant.cpp a **faithful reference implementation** of the ICLR 2026 paper
4. Provide a unique, defensible differentiation that no other C inference engine has

This is the **single highest-impact improvement** possible for the project.
