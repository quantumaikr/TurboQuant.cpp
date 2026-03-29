# TurboQuant.cpp — Session State

**Last updated**: 2026-03-29 (v0.9.1 non-matmul overhead optimization)
**Last commit**: pending

## Speed Progression
```
PyTorch CPU:        0.8 tok/s
v0.8 FP32:          5   tok/s  (6x PyTorch)
v0.8 Q8+threads:   21   tok/s  (26x)
v0.9 Q4+threads:   38   tok/s  (48x)
v0.9.1 optimized:  ??   tok/s  ← measure after this change
llama.cpp Q4_K_M:  ~50   tok/s  ← target
```

## What Works
- All 19 tests pass, zero warnings
- Q4 weights: 270 MB, Q8: 533 MB (vs 2.1 GB FP32)
- Self-contained C inference engine, 0 dependencies
- DeltaNet + Self-Attention hybrid forward pass
- KV cache quantization (Q4, 7.5x compression)
- Integer Q4×Q8 attention

## v0.9.1 Changes — Non-matmul Overhead Optimization

### Strategy A: NEON-optimized DeltaNet inner loops
- Fused decay + sk computation in a single NEON pass over state rows
- NEON outer product (S += outer(K, d)) fused with output (o = S @ Q)
- Eliminates 3 separate passes over dk×dv state matrix → 2 passes
- NEON L2 normalize with vectorized sum-of-squares and scaling
- NEON group norm (RMSNorm sum-of-squares)
- NEON swish(z) gate with fast_expf

### Strategy B: Batched conv1d + SiLU
- Combined conv1d + SiLU into single `causal_conv1d_silu_batch()`
- Specialized path for conv_width=4: unrolled dot product (no loop)
- Processes 4 channels together with NEON SiLU
- Eliminates per-channel function call overhead (6144 calls → 1536)

### Strategy C: Cached Q8 activation quantization
- Added `tq_matmul_q4_preq()` — takes pre-quantized int8 activation
- DeltaNet: quantize xb once, reuse for 4 Q4 matmuls (QKV, Z, A, B)
  - Saves 3× tq_quantize_row_q8 + 3× malloc/free per DeltaNet layer
  - 18 DeltaNet layers × 3 saved = 54 redundant quantizations eliminated
- Self-attention: quantize xb once, reuse for Q, K, V projections
  - Saves 2× quantization per self-attn layer
  - 6 self-attn layers × 2 saved = 12 redundant quantizations eliminated
- FFN: quantize xb once, reuse for gate + up projections
  - Saves 1× quantization per layer (all 24 layers)
  - 24 layers × 1 saved = 24 redundant quantizations eliminated
- Total: ~90 redundant Q8 quantizations eliminated per token

### Strategy D: Fast exp approximation
- `fast_expf()` using Schraudolph's algorithm (~6x faster than expf)
- Applied to: sigmoid in beta, softplus in gate, decay exp(gate), SiLU
- Kept precise expf() only for model parameters (A_log) that need accuracy
- Clamped to avoid overflow/underflow (|x| > 20 fallback)

### Files Modified
- `src/engine/tq_transformer.c` — All 4 strategies
- `src/engine/tq_ops.c` — Added tq_matmul_q4_preq(), fixed unused var warning
- `include/turboquant/tq_engine.h` — Added tq_matmul_q4_preq() declaration

## What Needs Work
1. Measure actual speed improvement (need model file for tq_run)
2. Q4 quality on short prompts
3. Metal GPU inference
4. More model architectures
