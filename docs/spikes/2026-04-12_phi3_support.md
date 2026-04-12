# Spike — Phi-3 / Phi-3.5 architecture support

**Date**: 2026-04-12
**Driver**: External user feedback (`docs/feedback/2026-04-12_0900.md`, item 2.6)
**Status**: Investigation complete; implementation gated on having a real GGUF to validate against
**Recommendation**: do NOT merge a fix without an end-to-end validation run

## Why Phi-3 matters

Phi-3.5-mini is the highest-value model NOT supported by quant.cpp:

- **vocab 32K** — smaller than SmolLM2 (49K), Llama-3.2-1B (128K), Gemma (256K)
- **3.8B params** — bigger than SmolLM2-1.7B but the small vocab keeps lm_head fast
- the tester estimated `~94 tok/s` (`60 tokens / 0.85 s`) before realizing the inference was producing garbage — that number reflects what the matmul kernels can do; only the attention path is broken

If we get this working, Phi-3.5-mini becomes the new "best speed/quality" recommendation, ahead of SmolLM2-1.7B.

## Current state

`tq_load_gguf` (in `quant.h`, lines 11640-11680) looks for these tensor names per layer:

```
blk.N.attn_q.weight    ← required to mark layer as self_attn
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_output.weight
```

When loading a Phi-3 GGUF, none of these exist — Phi-3 ships fused QKV. Phi-3's tensors (in llama.cpp's GGUF naming convention) are:

```
blk.N.attn_qkv.weight    ← shape [3 * hidden_dim, hidden_dim], fused
blk.N.attn_output.weight
blk.N.ffn_up.weight      ← may also be fused as ffn_up_gate, depending on converter
blk.N.ffn_down.weight
```

Result: `is_attn_layer = 0` for every layer, `n_attn_layers = 0`, the new hard-fail check in P0-B catches it and returns NULL with a clear error. No more garbage tokens — but no working inference either.

## Two implementation strategies

### Option A — Loader splits at load time

After detecting `attn_qkv`, dequantize the fused tensor, slice along the output dimension into three `[hidden_dim, hidden_dim]` views, re-quantize each as a separate Q4_K (or whichever type the GGUF used), and store them in `gguf_wq`/`gguf_wk`/`gguf_wv`.

**Pros**: zero forward-path changes, drops into existing `tq_matmul_gguf` calls.
**Cons**:
1. Doubles RAM during load (need both fused + split versions)
2. Re-quantization is **lossy** — running the original model through Q4_K → FP32 → Q4_K introduces measurable error
3. Won't work for tensor types we don't have a quantizer for (we'd need a quantizer for every supported GGUF type)
4. Slow at load

### Option B — Forward path dispatches fused matmul (RECOMMENDED)

Add a new field `gguf_wqkv` (data + type) to `tq_layer_weights_t`. Loader sets it from `blk.N.attn_qkv.weight` directly. Forward path checks: if `gguf_wqkv` is set, do one big matmul into a temp buffer of size `3 * hidden_dim`, then split into the existing `s->q`, `s->k`, `s->v` outputs.

**Pros**:
1. No re-quantization, no precision loss
2. No extra load-time work
3. Works with any GGUF type we already support in `tq_matmul_gguf`
4. Single big matmul is faster than 3 smaller ones (better cache reuse)

**Cons**:
1. Need a temp buffer for the fused output
2. New branch in the forward path (small)
3. Need to pass `q_dim`, `k_dim`, `v_dim` so the split knows where K starts and V starts (Phi-3 may not use GQA, but we can't assume)

`tq_matmul_gguf` already accepts `(weight, type, out_dim, in_dim)` — it doesn't care whether the underlying tensor is fused or not. We can call it once with `out_dim = q_dim + k_dim + v_dim`.

## Inspection results (2026-04-12)

Used `tools/gguf_inspect.c` against `bartowski/Phi-3.5-mini-instruct-Q4_K_M.gguf` (2.39 GB). Findings:

### Per-layer tensors (32 layers, 6 tensors each)

```
blk.N.attn_norm.weight    F32   [3072]
blk.N.attn_qkv.weight     Q5_K  [3072, 9216]    ← FUSED QKV (3 * 3072)
blk.N.attn_output.weight  Q4_K  [3072, 3072]
blk.N.ffn_norm.weight     F32   [3072]
blk.N.ffn_up.weight       Q4_K  [3072, 16384]   ← FUSED gate+up (2 * 8192)
blk.N.ffn_down.weight     Q6_K  [8192, 3072]
```

### Global tensors

```
token_embd.weight              Q4_K  [3072, 32064]
output.weight                  Q6_K  [3072, 32064]
output_norm.weight             F32   [3072]
rope_factors_long.weight       F32   [48]      ← LongRoPE
rope_factors_short.weight      F32   [48]      ← LongRoPE
```

### Metadata

- arch: `phi3`
- embedding_length: 3072 (hidden_dim)
- block_count: 32
- head_count: 32
- head_count_kv: 32 (NO GQA)
- rope.dimension_count: 96 (head_dim per head)
- rope.freq_base: 10000
- rope.scaling.original_context_length: 4096 (LongRoPE switch point)
- rope.scaling.attn_factor: 1.19024 (Q/K magnitude scaling for long context)
- context_length: 131072
- feed_forward_length: 8192
- vocab_size: 32064
- bos_token_id: 1, eos_token_id: 32000

### Conclusions

1. **Fused QKV** confirmed. Layout `[Q | K | V]` along output axis. Each section is `hidden_dim = 3072` floats. Total `9216 = 3 * 3072`.
2. **Fused FFN** ALSO confirmed. `ffn_up.weight` is `[hidden, 2*ff]` not `[hidden, ff]`. Layout `[?, ?]` — order TBD by validation, but llama.cpp's reference loads as `[gate, up]` chunked from this single tensor.
3. **LongRoPE present**: separate `rope_factors_short` and `rope_factors_long` tables of size 48 = head_dim/2. Used to rescale per-frequency RoPE rotations for sequences past the 4096-token original context.
4. **No special tokens for ChatML**. Phi-3 uses `<|user|>`, `<|assistant|>`, `<|end|>` (text strings, not BPE special tokens). Chat template differs from Llama-3 / ChatML.
5. **Vocab 32K** confirms the speed advantage — `lm_head` matmul is `3072 × 32064` vs Llama-3.2-1B's `2048 × 128256`. About 2.7× smaller per-token cost.

## What's still unknown (resolved by trial)

I need a real Phi-3 GGUF to verify:

1. **Exact tensor names**. llama.cpp's GGUF converter has changed conventions over the years. The fused tensor might be named:
   - `blk.N.attn_qkv.weight`
   - `blk.N.attn_qkv_proj.weight`
   - `blk.N.qkv.weight`
   - …and there may be a separate bias tensor

2. **Shape ordering**. Is the fused tensor `[Q | K | V]` along axis 0, or some other layout? Phi-3 has `n_heads = 32` and `n_kv_heads = 32` (no GQA in the 3.8B variant), so all three sub-tensors are the same size — but I want to verify.

3. **FFN fusion**. Does this Phi-3 GGUF use `ffn_up` + `ffn_gate` as separate tensors (llama-style) or `ffn_up_gate` (Phi-style fused)? If the latter, we have a second fused-tensor problem to solve in the same PR.

4. **RoPE config**. Phi-3 long-context variants use LongRoPE with two scaling factors (`short_factor`, `long_factor`). Phi-3-mini's 4K context might use vanilla RoPE — but Phi-3.5-mini's 128K context definitely uses LongRoPE. We'd need to read these from GGUF metadata and add them to `tq_rope`.

5. **Sliding window**. Phi-3 uses `n_block_sparse_window` (varies by layer in some variants). Whether the `mini` variant uses it is unclear.

6. **Special tokens**. Phi-3 uses `<|user|>`, `<|assistant|>`, `<|end|>` instead of ChatML — the chat template needs to know.

## Estimated effort once we have a GGUF

| Step | Effort |
|---|---|
| Tensor name detection (`attn_qkv` + variants) | XS — 20 lines |
| `gguf_wqkv` field + forward dispatch | S — 60 lines |
| `ffn_up_gate` if needed | S — 40 lines |
| LongRoPE if Phi-3.5-mini | M — 100-150 lines, needs careful validation |
| Sliding window detection | S — 30 lines (we have the infrastructure for Gemma) |
| Phi-3 chat template in `cli.py` | XS — 10 lines |
| Validation: load + 100 tokens + manual quality check | M — needs the GGUF |

**Total**: maybe 300-400 lines of focused code. Most of it is mechanical once we know the exact names.

## Recommendation

**Option B**, but only after one of:

1. **Tester provides** the exact Phi-3.5-mini-instruct-Q8 GGUF they used. Best path — same file the user already has running.
2. **Tester runs** a small inspector script we provide that dumps tensor names + shapes from their GGUF, so we can validate our assumptions without shipping the file.
3. **We pick** a specific bartowski Phi-3.5-mini Q4_K_M variant ourselves, download it, dump tensor names, and proceed. This is the slowest path because the failure modes (LongRoPE, sliding window) are subtle and easy to miss without ground-truth output to compare.

Until then: do NOT implement. The hard-fail in P0-B is the right transition state — users see a clear error and know to wait, instead of debugging garbage.

## Open questions for the human

1. Do we have access to the same Phi-3.5-mini GGUF the tester used? (`Phi-3.5-mini-instruct-Q8_0.gguf`, 3.9 GB)
2. If not, are we OK downloading one and using it as the reference? Storage / bandwidth?
3. Should I write the GGUF inspector script (path 2) so the tester can run it for us?
