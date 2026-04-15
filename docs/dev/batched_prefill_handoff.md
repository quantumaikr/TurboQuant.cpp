# Batched Prefill — Implementation Handoff (2026-04-15)

## Status

- ✅ **Strategy**: documented in `bench/results/2026-04-15_accelerate_gemm_microbench.md`.
  Apple AMX delivers 1.2 TFLOPS via cblas_sgemm; GEMV path peaks at ~15 GFLOPS.
  100× speedup is real but requires the workload to be batched.
- ✅ **Primitive**: `tq_batched_matmul_q4()` in `src/engine/tq_ops.c`.
  Unit-tested in `tools/test_batched_matmul.c` — 12/12 PASS, max_rel=0.0000,
  observed speedups 1.2-3.0× across realistic shapes (Phi-3.5, Llama 3.x).
- ✅ **Integration scaffolding**: `tq_forward_batch()` in `src/engine/tq_transformer.c`
  + opt-in flag `TQ_BATCH_PREFILL=1` in `src/engine/tq_generate.c`.
  Compiles, runs, falls back gracefully on unsupported architectures.
- ❌ **Numerical correctness of tq_forward_batch**: end-to-end output diverges
  from baseline. Matmul primitive is bit-identical (verified at primitive
  level), so the bug is somewhere in the surrounding orchestration (state,
  RoPE, KV cache layout, residual flow, or embedding source).

Reproduce divergence:
```bash
DYLD_LIBRARY_PATH=build TQ_BATCH_PREFILL=1 \
  build/quant models/Llama-3.2-1B-Instruct-Q8_0.gguf -p "Hello world" -n 5 -T 0
# prints: " hell hel hell h hel"

DYLD_LIBRARY_PATH=build \
  build/quant models/Llama-3.2-1B-Instruct-Q8_0.gguf -p "Hello world" -n 5 -T 0
# prints: " I'm so excited"  ← baseline (correct)
```

## Session 2 findings (2026-04-16 early AM)

After extensive layer-by-layer diff between batched and per-token paths:

**What's bit-identical**:
- tok0 (pos=0) through Layer 15 — every sub-op, every layer
- tok1 (pos=1) through Layer 2 final Xres
- Layer 3 attention-residual value at indices 0, 2, 3, 4 (partial match)

**What diverges**:
- Layer 3 tok1 attention-residual at indices 1, 5, 6, 7 — exactly 1 ULP off
- This 1-ULP drift compounds ~1%/layer → wrong token by Layer 15

**Surprising**: Setting `TQ_BATCHED_SERIAL=1` (which replaces my bm_q4_worker
with literal per-token `tq_matmul_q4_preq` calls — the SAME function baseline
uses) STILL produces the divergence. So the bug is not in the batched matmul
accumulator order; it's somewhere in the broader orchestration of
tq_forward_batch when processing multi-token.

**Fixed along the way** (each moved Layer 0 closer to bit-identical):
- Q8 quantization: `roundf` → `tq_quantize_row_q8` (NEON RNE)
- FP16 conversion (write): inline → `f32_to_fp16_vec`
- FP16 conversion (read): inline → NEON `vcvt_f32_f16`
- Attention score accumulation: scalar → `vfmaq_f32` NEON
- bm_q4_worker: scalar accumulator → NEON `float32x4_t sumv[]` + `vaddvq_f32`

**Remaining hypothesis** (to test next session):
The drift is at specific indices, not random. Index 1 of Layer 3 tok1 diverges
but indices 0, 2, 3 don't. This is consistent with a SPECIFIC memory location
being read slightly off. Possibilities:
- Aliasing: my X buffer might be accidentally read before fully written in
  some multi-token iteration (out-of-order thread effect)
- FP16 round-trip on a specific value whose LSB happens to sit on a boundary
- The `tq_forward` final call (after batched) reads K-cache positions [0..pos-1]
  written by batched; if ANY of those are 1 ULP off for any layer, final
  attention sees slightly wrong history. Could be compounding effect.

**Concrete next-session experiment**:
1. Dump Layer 3 tok0 wo matmul output (OB→X) byte-for-byte vs baseline
2. Dump Layer 3 tok1 attention scores (att[0], att[1]) vs baseline
3. If scores differ, dump K-cache at layer 3 pos=0 vs baseline
4. If K-cache differs, dump the WK matmul output for tok0 at layer 3

## Latest session findings (2026-04-15 evening)

- ✅ **SANITY mode confirms orchestration is correct**. Setting
  `TQ_BATCH_SANITY=1` makes `tq_forward_batch` simply call `tq_forward`
  N times and the output matches baseline ("I'm so excited"). The bug
  is purely in the per-token unrolled batched code, not in the integration
  with `tq_generate`.

- ✅ **Q4 matmul primitive verified at runtime** with both bias and Q-norm
  fixes added (NULL for Llama anyway). Q2 residual handling added too —
  but Llama 3.2 1B's load-time Q4 conversion does NOT produce Q2
  residuals (`wq_q2 == NULL` confirmed by debug print), so Q2 isn't the
  culprit either.

- ❌ **Bug still present in actual batched path**. Output remains
  "hell hel hell..." at N=2.

- 🔍 **Strong suspect**: tq_forward uses different matmul function
  variants for different projections (`tq_matmul_q4q2_preq` for wq/wk/wv,
  `tq_matmul_q4` for wo/gate/up/down). Although they should be
  functionally equivalent, there may be subtle differences (e.g.,
  rounding mode in input quantization, scale normalization). The
  systematic next step is to **dump s->x[0..3] after layer 0 from both
  paths** — this isolates which sub-op diverges.

## Debugging plan for next session

1. **Add intermediate-state dumps** to `tq_forward` and `tq_forward_batch`.
   Compare layer-0 outputs (Xres after attention residual) byte-by-byte
   for the same single token. If they differ, the bug is at layer 0
   before any batching matters.

2. **Likely suspects ranked by probability**:

   **(a) RMSNorm input vs output buffer.** My code calls
   `tq_rmsnorm(XBN+n*dim, Xres+n*dim, ...)`. tq_forward calls
   `tq_rmsnorm(s->xb, s->x, ...)`. The pattern is the same, but verify
   the eps value is exactly `c->rms_norm_eps` and the weight pointer is
   `layer->attn_norm` (not `layer->ffn_norm`).

   **(b) KV cache stride.** Forward uses `cache_kv_dim` (computed via
   sliding/full max). For Llama 3.x non-Gemma this should equal kv_dim,
   but worth printing both at write time to confirm.

   **(c) attn_output_gate.** Llama doesn't have it, but verify
   `c->attn_output_gate == 0`.

   **(d) Output deinterleave.** When attn_output_gate is set, Q is
   interleaved with a gate. We don't handle this in tq_forward_batch
   because we bail when the flag is set... but is the bail check there?
   (Currently no — should add.)

3. **RoPE freq formula.** My batched code:
   ```c
   float base = 1.0f / powf(c->rope_freq_base, 2.0f * i / (float)c->head_dim);
   float freq = base / model->rope_freqs[i];
   ```
   Compare to tq_forward line 1217-1219:
   ```c
   float base_freq = 1.0f / powf(rope_base, 2.0f * i / (float)rope_n_dims);
   float freq = base_freq / model->rope_freqs[i];
   ```
   Note: `rope_n_dims` may not equal `head_dim`! For Gemma 4 they differ.
   For Llama 3 should be same but verify `c->rope_n_dims` and use it
   instead of head_dim.

4. **The attention computation**. Mine uses the simplest causal scan over
   K/V cache positions. tq_forward uses the same but might apply scaling
   factors (logit_softcap for Gemma, attention_bias for some, etc.).
   Llama 3 should be plain — verify no scale factor is missed.

## Architectural targets

Once correctness is achieved, expected gains (per microbench):
- N=8 prefill chunk: ~3× per-matmul vs single
- N=32 prefill chunk: ~30× per-matmul vs single
- N=128 prefill chunk: ~60-100× per-matmul vs single

The `tq_batched_matmul_q4` primitive currently gives 1.5-3× because it
works over Q4 weights and dequant overhead caps the win below the AMX
ceiling. Future optimizations (in priority):

1. **Persistent FP16 lm_head** — ~1.5× decode + huge prefill win for Qwen3.5-4B
2. **BNNS quantized GEMM** — direct AMX with int8 weights, no dequant overhead
3. **MPSGraph for multi-layer fused forward** — entire layer on GPU

## Out-of-scope for this implementation

Don't try to make tq_forward_batch handle every architecture in v1. Bail
on:
- `is_gemma4`, `is_moe`, `has_fused_qkv`, `has_fused_up_gate`
- `n_kv_shared_layers > 0`
- Any layer with `delta_a_log` (DeltaNet)
- `attn_output_gate` (rare; just bail)
- `partial_rotary_factor > 0` (Phi-3 LongRoPE)

The fast path should cover Llama 1B/3B/8B Q8_0 and Q4_K_M with load-time
Q4 conversion. Other models stay on the per-token path. We can extend
case-by-case.

## Files touched

- `include/turboquant/tq_engine.h` — declared `tq_forward_batch` and
  `tq_batched_matmul_q4`.
- `src/engine/tq_ops.c` — implemented `tq_batched_matmul_q4` and worker.
- `src/engine/tq_transformer.c` — implemented `tq_forward_batch` (WIP).
- `src/engine/tq_generate.c` — gated integration behind TQ_BATCH_PREFILL.
- `tools/test_batched_matmul.c` — primitive correctness + speed test.
- `bench/results/2026-04-15_accelerate_gemm_microbench.md` — strategy doc.
