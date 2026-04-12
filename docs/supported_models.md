# Supported Models

quant.cpp loads GGUF files from HuggingFace, but only some model
architectures are fully wired through the inference path. This page
tracks what works, what loads-but-fails, and how to pick a model.

## TL;DR — Recommended models

| Use case | Model | Why |
|---|---|---|
| **First-time install** | `SmolLM2-1.7B` (Q8) | Fastest end-to-end on a laptop. Vocab 49K keeps the lm_head matmul small (~12 tok/s on Apple M3). |
| Smaller download | `Llama-3.2-1B` (Q4_K_M) | 750 MB vs 1.7 GB, but ~5x slower at inference time due to 128K vocab. |
| Quick smoke test | `SmolLM2-135M` (Q8) | 138 MB download to verify the install path. Output quality is poor — not for real use. |

```bash
# CLI quickstart
quantcpp run smollm2          # SmolLM2-1.7B (recommended)
quantcpp run smollm2:135m     # SmolLM2-135M (smoke test only)
quantcpp run llama3.2:1b      # smaller download, slower
```

```python
# Python quickstart
from quantcpp import Model
m = Model.from_pretrained("SmolLM2-1.7B")
print(m.ask("What is gravity?"))
```

## Architecture compatibility matrix

| Architecture | GGUF Load | Tokenizer | Attention | Inference | Status |
|---|:---:|:---:|:---:|:---:|---|
| **llama** (SmolLM2, Llama-3.x, Mistral) | ✅ | ✅ | ✅ | ✅ | **Fully supported** |
| llama with 128K vocab (Llama-3.2-1B) | ✅ | ✅ | ✅ | slow | Supported, vocab is the bottleneck |
| **gemma** (Gemma 2) | ✅ | ✅ | ✅ | ✅ | Supported |
| **gemma3** | ✅ | ✅ | ✅ | ✅ | Supported with hybrid sliding-window attention |
| **gemma4** (Gemma-4-E2B / E4B) | ✅ | ✅ | ⚠️ | ⚠️ | Partial — some Q4_K_M variants produce garbage; report with file SHA256 |
| **qwen** / **qwen2** | ✅ | ✅ | ✅ | ✅ | Supported |
| **qwen3.5** (DeltaNet hybrid) | ✅ | ✅ | partial | ⚠️ | Partial — pure-attention layers work, DeltaNet hybrid still being validated |
| **phi3** / **phi3.5** (fused QKV) | ❌ | — | — | — | **Not supported** — uses `attn_qkv`, see "Why phi3 is hard" below |

✅ = works · ⚠️ = loads but inference is unreliable · ❌ = load fails fast with a clear error (since 2026-04-12)

If you load an unsupported architecture, the loader now prints:

```
tq_load_gguf: ERROR — model architecture 'phi3' is not supported.
  Detected 0 self_attn layers and no DeltaNet weights.
  This usually means the model uses fused QKV projection
  (e.g., Phi-3 `attn_qkv`) which quant.cpp does not yet handle.
  See docs/supported_models.md for the architecture support matrix.
```

…and `tq_load_gguf` returns NULL, so callers can fail-fast instead of
silently producing garbage tokens.

## Why vocab size dominates speed

quant.cpp generates one token at a time. Every token requires a
`lm_head` matmul of shape `[hidden_dim, vocab_size]`. For a typical 1B
model with `hidden_dim = 2048`:

| Model | vocab_size | lm_head FLOPs/token |
|---|---:|---:|
| SmolLM2-1.7B | 49,152 | 100 M |
| Llama-3.2-1B | 128,256 | 263 M |

Llama-3.2-1B has fewer parameters (1.0B vs 1.7B) but its lm_head matmul
is 2.6x bigger, and on CPU it dominates wall time. External user
benchmarks on Apple M3 (8-core CPU, 16 GB RAM):

| Model | tok/s | 60-token latency |
|---|---:|---:|
| SmolLM2-1.7B (Q8, vocab 49K) | ~12.5 | ~5 s |
| Llama-3.2-1B (Q4_K_M, vocab 128K) | ~2.3 | ~27 s |

**Take-away**: when picking a model for an embedded / laptop scenario,
vocab size is a better predictor of interactive latency than parameter
count. Pick the smallest vocab that produces output you're happy with.

## Why phi3 is hard

Phi-3 / Phi-3.5 uses a *fused* QKV projection: instead of three separate
tensors `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, it ships one
`attn_qkv.weight` with all three projections concatenated along the
output dimension.

quant.cpp's GGUF loader currently looks for the three-tensor layout
(`blk.N.attn_q.weight` etc.). When it loads a Phi-3 GGUF, none of those
names match → 0 self_attn layers detected → forward pass runs against
zero-initialized attention weights → garbage tokens.

Adding Phi-3 support requires either:

1. **Loader splits** `attn_qkv.weight` into the three views at load time
   and writes them into the existing `wq`/`wk`/`wv` slots, OR
2. **Forward path** learns to dispatch a fused QKV matmul when the
   loader detects the fused tensor.

Option (1) is simpler but doubles the working set during load. Option
(2) is the right long-term answer. There's a tracking issue / spike in
progress; until then Phi-3 is the highest-value missing architecture for
quant.cpp's "speed + quality" target (Phi-3.5-mini has vocab 32K plus
3.8B params — it would beat both SmolLM2-1.7B and Llama-3.2-1B at
interactive use).

## Reporting an unsupported model

If you tried a model that's not in the matrix above, please open an
issue with:

- The HuggingFace repo + filename
- The exact `tq_load_gguf:` log lines (including `architecture = '...'`)
- The first ~50 generated tokens (so we can see whether it's garbage,
  partial garbage, or just wrong-language)

Don't include the model file itself — link to the HuggingFace page.
