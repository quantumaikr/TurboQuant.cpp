# Reddit r/LocalLLM Response Drafts (2026-04-03)

Post: 16 upvotes, 5.4K views, 19 comments

---

## @MrRandom04 — "re-implementing all of llama.cpp just to add whatever approach"

We don't intend to replace llama.cpp. We have a self-contained llama.cpp integration patch (`integrations/llamacpp/patch/`, 4 files, ~1000 lines) that adds `--cache-type-k tq_kv_1b` as a drop-in option. The standalone engine exists for research and to verify the algorithm on multiple architectures (Llama, Gemma, Qwen, Qwen-MoE — 4 verified). The goal is to get TurboQuant KV into llama.cpp as a native cache type.

---

## @dinerburgeryum — "codebook calibration sensitive to out-of-domain data?"

Good question. The **1-bit path doesn't use a codebook at all** — it's just `sign(RHT(key))`, so there's nothing to calibrate and nothing domain-sensitive. The RHT seed is fixed per-block and model-independent.

The codebook is only used for 3-bit and 4-bit modes (Lloyd-Max optimal for N(0,1)). Our `--calibrate` tool showed 49.7% MSE improvement with model-specific codebooks, but the 1-bit path skips all of this.

---

## @Viper-Reflex — "does this make my 24GB 3090 run bigger models?"

KV compression helps most with **long contexts**, not bigger models. With 1-bit K + Q4 V, KV memory drops ~5x. For a 27B model at 32K context:
- Before: ~2.5 GB KV cache
- After: ~500 MB KV cache → frees ~2 GB for longer context or larger batch

If you're already fitting a model in 24GB, TurboQuant lets you push context from 32K → 100K+ on the same hardware. But it won't help you fit a model that's too large for VRAM (weight memory is separate from KV cache).

Note: we currently don't have CUDA GPU acceleration (it compiles but is untested). That's next on the roadmap.

---

## @Blizado — "zero quality loss claim" (already responded)

Updated README: "almost no quality loss (PPL +0.03%)".

Clarification:
- K-only (V as FP16): PPL is exactly +0.00% — measured identical on both Gemma 4B and SmolLM2 1.7B (Llama arch)
- K + Q4 V: PPL +0.03% — near-zero, not zero
- "byte-identical" refers to greedy decoding up to ~100 tokens, not infinite sequences

---

## Key takeaway from Reddit feedback

1. **"zero quality loss" was overstated** → fixed to "almost no" with exact PPL
2. **"why not just integrate into llama.cpp?"** → we have a patch, that's the plan
3. **Technical curiosity is high** — 5.4K views, people want to understand the math
4. **Skepticism is healthy** — the Blizado/No-Manufacturer criticism pushed us to be more precise
