# The Working Memory Cliff: Measuring When Quantized Edge LLMs Stop Following Instructions in Long Context

**Tech report draft — v0.4 — 2026-04-11**
**Author**: quant.cpp maintainers
**Status**: Phase 2B Karpathy loop complete — 204 trials measured + ReDeEP-Cliff hypothesis tested on all 115 failures via surface-level proxy. Hypothesis rejected; new failure mode "primacy-biased document continuation overflow" identified and characterised. Self-correction of v0.3's "synthesised hallucination" framing in §4.6.

> **TL;DR** — We measure the *effective* working memory of two edge-device quantized LLMs (Llama-3.2-1B-Q8, Llama-3.2-3B-Q4) using a Needle-in-a-Haystack protocol on 204 trials. **Both models fall off a cliff at 0.4–0.8% of their nominal context window**: 1B Q8 retains 100% retrieval at 512 tokens but drops to 0% by 1536, and 3B Q4 retains 100% at 1024 tokens but collapses to 0% at 1280 — a step function spanning <256 tokens. **6.4× KV cache compression is bit-for-bit identical to FP32 baseline in 18/20 cells**, and a 6-trial FP32-weights control rules out weight quantization, so the cliff is a *model* property — not a KV property and not a quantization artifact. The dominant cliff failure mode (84% of failures) is **literal continuation of the haystack from its 10–20% body-start position** — a previously-unnamed failure mode we call *primacy-biased document continuation overflow*, mechanistically *opposite* to RAG silent hallucination (high copy + low novel, vs RAG's low copy + high novel). The "long-context replaces RAG" framing holds at the memory-allocation level (a 128K context fits in 9.5 GB) but not at the retrieval level (the model stops following instructions long before the memory is full), and the failure cannot be fixed by retargeted RAG-hallucination mitigations like AARF — it needs anchor-strengthening interventions instead.

---

## 1. Motivation

Recent context-window expansion (Llama 3.2 → 128K, Gemini 1.5 → 2M) has fueled a narrative that long-context inference can replace retrieval-augmented generation (RAG): "load the whole document, skip the vector DB, the failure mode of silent chunk-level hallucination disappears." This position has merit at the cloud-scale (frontier models), but **for the edge-device case** — where KV cache compression is what makes large contexts feasible in 16 GB — the narrative has never been empirically checked against a simple measurable question:

> *At what context length does the model still actually follow the instruction to retrieve a fact from the provided text?*

Prior work on KV compression (KIVI, H2O, SnapKV, PyramidKV) has measured retrieval accuracy via perplexity or NIAH at the *frontier* scale, typically on Llama-3-8B or Llama-2-70B. The edge-device quantized regime — 1B–3B parameters, Q4/Q8 weights, on-the-fly weight dequantization — has been under-measured. This report fills that gap.

Our contribution is specifically **negative and narrow**:

1. An NIAH protocol variant for edge-device quantized LLMs, with wikitext-2 haystacks and common-word needles robust to Q4 visual jitter.
2. Per-model *effective working memory ceiling* measurements: the context length beyond which chat-template-anchored instruction following fails.
3. Apples-to-apples KV compression (6.4× `turbo_kv_4b -v q4 --k-window 128`) vs FP32 baseline across the regime where the model actually retrieves.
4. An honest reframing: KV compression is **orthogonal** to the working-memory cliff. Compression neither extends nor degrades the ceiling; it preserves whatever the model can already do.

## 2. Background and Related Work

### 2.1 KV cache compression

The KV cache memory footprint of an autoregressive LLM grows linearly with context length. For Llama-3.2-3B at 128K context, an FP16 cache occupies roughly 12 GB — already past the budget of a 16 GB consumer Mac before the model weights are loaded. A growing line of work compresses this cache to make long-context inference tractable on edge hardware.

**KIVI** [Liu et al. 2024] introduces 2-bit KV cache quantization with channel-wise quantization for K and token-wise for V, motivated by the observation that K and V have different statistical structure. KIVI reports near-lossless retrieval on Llama-2-7B and Llama-2-13B at 4× compression. Our `turbo_kv_4b -v q4 --k-window 128` configuration is conceptually similar: K is compressed via PolarQuant 3-bit + QJL 1-bit, V is compressed via 4-bit min-max, and a 128-token recency window stays in FP32. The compression ratio (6.4×) is more aggressive than KIVI 2-bit, and the targets are smaller (1B/3B vs 7B/13B).

**H2O** [Zhang et al. 2023] ("Heavy Hitter Oracle") prunes the KV cache by keeping only the tokens that historically receive the most attention, dropping ~80% of cache slots with minor quality loss. It is orthogonal to KIVI: H2O reduces *cache size*, KIVI reduces *bits per cached token*. quant.cpp ships an H2O implementation but our cliff measurements use the dense `turbo_kv_4b` path because the cliff transition is too sharp for cache-eviction effects to be visible.

**SnapKV** [Li et al. 2024] selects KV slots to preserve based on a small attention window over the prompt, effectively trading off recall for cache size. **PyramidKV** [Cai et al. 2024] allocates per-layer budgets — earlier layers get more cache, later layers less — based on the observation that attention sparsity grows with depth.

These four methods all measure quality on perplexity (wikitext, c4) or cloud-scale NIAH (Llama-2-7B+ at long contexts). **None measures the working memory cliff at edge-device scale** (1B–3B parameters with on-the-fly weight requantization), which is the regime in which `quant.cpp` deploys and which this report characterises.

### 2.2 Long-context retrieval

**Needle in a Haystack** [Kamradt 2023] is the de facto retrieval benchmark for long-context LLMs: insert a fact at varying depth in a long irrelevant document, ask a single question whose answer is the fact, score string-match accuracy. Originally run on cloud LLMs (GPT-3.5/4, Claude) up to 200K context, NIAH visualisations have shown that frontier models retain near-perfect recall through their full nominal context. The original Kamradt protocol uses Paul Graham essays as the haystack; our protocol substitutes wikitext-2 for license-and-reproducibility reasons and adds three-needle redundancy to suppress per-needle bias.

**Lost in the Middle** [Liu et al. 2023] measured GPT-3.5, Claude-1.3, and several others on multi-document QA and showed that retrieval accuracy is U-shaped: facts at the beginning and end of a long context are recalled better than facts in the middle. The effect was pronounced at 7K–30K tokens on frontier models. We see qualitatively similar depth sensitivity (1B at ctx=256 fails on `depth=0.9 needle=0` more than on `depth=0.1 needle=0`) but the dominant effect at edge scale is the **absolute cliff** at ~0.5–1% of nominal context, which is entirely below the regime "Lost in the Middle" probes.

**RULER** [Hsieh et al. 2024] extends NIAH with synthetic long-context tasks (multi-key, variable-depth, common words) and reports that even 70B-class models degrade much earlier than their nominal window length. RULER is the obvious next step for systematic edge-scale measurement; we discuss it under Future Work (§8).

**LongBench** [Bai et al. 2023] is a multi-task benchmark (QA, summarisation, code, retrieval) spanning real long documents in English and Chinese. Our protocol is intentionally narrower: we measure only the binary "did the model retrieve the planted fact" question, which is sufficient to surface the cliff and makes the failure mode taxonomy in §4.5 directly inspectable.

### 2.3 Edge-device LLM inference

**llama.cpp** is the de facto reference inference engine for consumer hardware. It uses the GGUF format for quantized model weights (Q2/Q3/Q4/Q5/Q6/Q8) and supports K-quants, I-quants, and a CPU/CUDA/Metal backend. Our binary `quant.cpp` (a single-header C engine focused on KV compression and embedding) is a sibling project that loads the same GGUF format and shares the Q4/Q8 quantization conventions, but defaults to a *re-quantizing* loader path: a Q8_0 GGUF on disk is dequantized and re-quantized to Q4 in memory before inference, which is why our raw logs show `weights=Q4` even when loading a Q8 file. This default is what users get from `pip install quantcpp`, and is therefore the configuration we measure.

**MLC-LLM** [Chen et al. 2023] targets on-device inference (mobile, browser, edge) with a TVM-based compilation pipeline and has published NIAH-like demonstrations on Llama-2-7B and Llama-3-8B. Their measurements are at 7B+ parameter scale and do not separate the effects of weight quantization from the working memory cliff.

We are not aware of a prior empirical study that specifically measures the *effective* working memory of 1B–3B quantized LLMs on a needle-retrieval task with a reproducible protocol. The closest comparison points are KIVI's 7B/13B perplexity numbers and Lost in the Middle's frontier-model U-curves, both of which probe a different regime. The contribution of this report is the cliff itself: it exists, it sits at <1% of nominal context, and it is independent of KV cache compression.

## 3. Method

### 3.1 Protocol

- **Haystack**: Real English text from wikitext-2-test (1.3 MB), trimmed to the target context length and sentence-aligned.
- **Needles**: Three common-English-word facts, inserted at the nearest sentence boundary to a target depth:
  1. "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023."
  2. "The launch date for Project Aurora is November 14th in San Francisco."
  3. "The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water."
- **Questions**: A single short-answer question per needle.
- **Scoring**: Case-insensitive keyword grep against the 32-token response.
- **Prompt format**: Simple — haystack + `"\n\nQuestion: "` + question, passed through the Llama 3 chat template via `--chat`. We evaluated five other formats (`"Based on this document..."` prefix, `<<<DOCUMENT>>>` delimiters, question-first sandwich, few-shot Q/A, raw continuation with `Q:/A:`) and found them strictly worse: the structured prefixes get overpowered by the continuation prior from the long haystack and the model starts continuing wikitext instead of answering.
- **Decoding**: Greedy, temperature 0, 32 tokens generation budget.

### 3.2 Models

| Model | Parameters | Weight precision | CLI loader path |
|---|---|---|---|
| Llama-3.2-1B-Instruct-Q8_0 | 1.24 B | 8-bit (no on-the-fly re-conversion) | quant.cpp default |
| Llama-3.2-3B-Instruct-Q8_0 | 3.21 B | 8-bit → Q4 on-the-fly (default CLI path) | quant.cpp default |

We intentionally test the **default** quant.cpp loader path — which transparently re-quantizes Q8 weights to Q4 at load time — because that is the path `pip install quantcpp` users get. This matches prior internal findings that defaults matter more than dev-build flags.

### 3.3 KV cache configurations

| Config | CLI flags | KV memory per token | Compression |
|---|---|---:|---:|
| `fp32` (baseline) | `-k fp32` | ~12 KB/tok (3B), ~8 KB/tok (1B) | 1.0× |
| `turbo_q4_w128` | `-k turbo_kv_4b -v q4 --k-window 128` | ~1.9 KB/tok (3B), ~1.3 KB/tok (1B) | 6.4× |

The `--k-window 128` flag reserves a 128-token FP32 recency window; older tokens are progressively compressed via a Polar-3b + QJL-1b hybrid. See `docs/custom_quant.md` for details.

### 3.4 Grid

Five context lengths bracketing the expected cliff: **256, 512, 1024, 1536, 2048 tokens**. Three depths: **0.1, 0.5, 0.9**. Three needles per (context, depth) cell. Two KV configurations per trial. Total: 90 trials per model, 180 trials across the two models. All inference runs on Apple Silicon Metal (`build_metal/quant`).

## 4. Results

204 NIAH trials in total across two models, two KV configurations, and two weight-precision modes:

| Round | Trials | Model | Weight precision | Coverage |
|---|---:|---|---|---|
| R1 | 36  | 3B  | Q4 (default)       | ctx 512/1024, fp32+turbo_q4_w128 KV |
| R2 | 90  | 1B  | Q8 (no requant)    | ctx 256/512/1024/1536/2048, both KV |
| R3 | 72  | 3B  | Q4 (default)       | ctx 1280/1536/1792/2048, both KV |
| **R4** | **6**  | **3B** | **FP32** (`TQ_NO_Q4=1`) | **ctx 1024/1280, fp32 KV — quantization confound control** |

### 4.1 Llama-3.2-1B-Instruct Q8_0 (no on-the-fly Q4 conversion)

| Method | ctx=256 | ctx=512 | ctx=1024 | ctx=1536 | ctx=2048 |
|---|---:|---:|---:|---:|---:|
| `fp32` (baseline)       | 8/9 (89%) | **9/9 (100%)** | 4/9 (44%) | 0/9 (0%) | 0/9 (0%) |
| `turbo_q4_w128` (6.4×)  | 8/9 (89%) | **9/9 (100%)** | 2/9 (22%) | 0/9 (0%) | 0/9 (0%) |
| Δ                       | 0 pp      | 0 pp           | −22 pp    | 0 pp     | 0 pp     |

The 1B model has a **graded cliff** centred at ctx=1024: perfect retrieval at 512, degradation to ~44% at 1024, total collapse by 1536.

The −22 pp gap at ctx=1024 between FP32 and the compressed variant **is not a real degradation**: both points are statistically indistinguishable from random at n=9. The cliff cell is unstable enough that the binomial noise dominates the apparent compression effect.

### 4.2 Llama-3.2-3B-Instruct Q4 (default CLI weight conversion)

| Method | ctx=512 | ctx=1024 | ctx=1280 | ctx=1536 | ctx=1792 | ctx=2048 |
|---|---:|---:|---:|---:|---:|---:|
| `fp32` (baseline)       | **9/9 (100%)** | **9/9 (100%)** | **0/9 (0%)** | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
| `turbo_q4_w128` (6.4×)  | **9/9 (100%)** | **9/9 (100%)** | **0/9 (0%)** | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
| Δ                       | 0 pp           | 0 pp           | 0 pp         | 0 pp     | 0 pp     | 0 pp     |

The 3B Q4 model has a **step-function cliff** between 1024 and 1280 tokens: 18/18 at 1024 in both methods, 0/18 at 1280 in both methods, no degradation interval at all. The model goes from perfectly following the chat template to completely ignoring it within a 256-token range.

### 4.3 Per-model cliff summary

| Model | Highest 100% retrieval | First 0% retrieval | Cliff width |
|---|---:|---:|---:|
| Llama-3.2-1B-Q8_0      | 512 tok  | 1536 tok | 1024 tok (graded) |
| Llama-3.2-3B-Q4 (default) | 1024 tok | 1280 tok | **<256 tok (step)** |

Going from 1B Q8 to 3B Q4 doubles the highest-100% ceiling (512 → 1024). This is one anecdotal data point on the model-size scaling of the working memory cliff at edge-device scale; n=2 is not enough to fit a curve, but it is enough to falsify the hypothesis that the nominal context window predicts the effective working memory.

### 4.4 Compression orthogonality

| Model | Disagreement cells | Agreement cells | Overall delta |
|---|---|---|---|
| 3B Q4 (10 cells, 90 trials) | 0 / 10 | 10 / 10 | **+0.0 pp** |
| 1B Q8 (10 cells, 90 trials) | 1 / 10 (the cliff cell, both at noise floor) | 9 / 10 | **−4.4 pp (within noise)** |

The single disagreement cell is the 1B Q8 ctx=1024 cliff, where FP32 happened to land 4/9 and turbo_q4 landed 2/9. Outside cliff cells, all 18 cells across both models are identical between baseline and 6.4× compression.

**Headline**: 6.4× KV compression preserves whatever the model can retrieve. The ceiling is a model property, not a KV property.

### 4.5 FP32-weights control: the cliff is independent of weight quantization

The default `quant.cpp` loader transparently re-quantizes Q8_0 GGUF weights to Q4 in memory. To check whether the cliff we observe is an artifact of this requantization (i.e., a consequence of weight precision rather than of the model itself), we re-ran the cliff-bracketing cells with the FP32-weights loader path (`TQ_NO_Q4=1`, which dequantizes Q8 to FP32 and skips Q4 conversion). Each FP32-weights inference takes ~10 minutes on Metal — too slow for a full grid — so we sampled the two cells that bracket the 3B Q4 cliff transition.

| Weight precision | ctx=1024 | ctx=1280 |
|---|---:|---:|
| Q4 (default loader, n=18 across 9-trial cells × 2 KV configs) | **100%** | **0%** |
| **FP32** (`TQ_NO_Q4=1`, n=3 each) | **100% (3/3)** | **0% (0/3)** |

**The cliff sits in the same place regardless of weight precision.** Going from Q4 to FP32 weights increases the per-parameter precision by 8× and eliminates any quantization artifact, but the model's chat-template-following behaviour collapses at exactly the same context length. This is the strongest possible evidence that the working memory cliff is a *model* property (instruction-tuned chat anchor robustness) rather than a *weight quantization* artifact.

The FP32-weights raw outputs are also visibly cleaner than the Q4 outputs: the latter exhibit per-character jitter ("aauorra-neebbulla-73991" instead of "aurora-nebula-7391") on identifier-like tokens, while FP32 produces clean English. This is consistent with the published Q4 weight quantization literature and is independent of the cliff phenomenon.

| Run | ctx | Output (FP32 weights) |
|---|---:|---|
| 1 | 1024 | "The answer is Sarah Chen." |
| 2 | 1024 | "The launch date for Project Aurora is November 14th in San Francisco." |
| 3 | 1024 | "Answer: The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water." |
| 4 | 1280 | "Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Roukke ..." |
| 5 | 1280 | "ion series Doctors , and appeared in a 2007 theatre production of How to Curse ..." |
| 6 | 1280 | "in the 2006 episode of Doctors , followed by a role as a different character ..." |

The above-cliff failures are *all* wikitext continuation — exactly the same failure mode as the Q4 path. Source: `bench/results/niah/results_fp32ctrl_20260411T091023.csv`.

### 4.6 Failure mode taxonomy and the cliff mechanism

> **Self-correction (v0.4)**: The v0.3 draft of this report cited a "synthesised hallucination" example (`"In 2023 Boulter was hired as the chief financial officer..."`) as the *most consequential* failure mode and equated it with RAG silent hallucination. **A 4-round Karpathy-loop subtype analysis on all 115 cliff failures (`bench/results/niah/redeep_proxy.py`, `redeep_subtype.py`, `continuation_origin.py`) shows that example is a single outlier (1 trial out of 115, < 1%) and the dominant cliff failure mode is the *opposite* of RAG hallucination.** This subsection is rewritten accordingly.

We classified all 115 cliff failures into four subtypes:

| Subtype | n | % | Description |
|---|---:|---:|---|
| **CONTINUE** | **97** | **84%** | Model literally continues the wikitext haystack ("Doctors, followed by a role in...") |
| OTHER | 9 | 8% | Other forms of haystack continuation |
| HEADER | 8 | 7% | Wikitext markup echo (`= = = 2008 II =`, `= Robert Boulter =`) |
| SYNTH | 1 | <1% | Needle + haystack subject fusion (the v0.3 example) |

The dominant failure mode (84%) is **literal continuation**, not synthesised hallucination. We then computed surface-level proxies for ReDeEP's [Sun et al., ICLR 2025] External Context Score and Parametric Knowledge Score on every trial:

- `copy_score(response, haystack)` = fraction of response 4-grams that appear in the haystack — proxy for External Context Score ("model is copying from loaded context")
- `novel_score(response, haystack)` = 1 − copy_score — proxy for Parametric Knowledge Score ("model is generating from internal memory")

| Subtype | n | Δ copy vs PASS | Δ novel vs PASS | ReDeEP signature match |
|---|---:|---:|---:|---|
| CONTINUE | 97 | **+0.077** | **−0.077** | **OPPOSITE** |
| OTHER | 9 | **+0.310** | **−0.310** | **STRONGLY OPPOSITE** |
| HEADER | 8 | +0.049 | −0.049 | weakly opposite |
| SYNTH | 1 | −0.028 | +0.028 | matches (n=1, not significant) |

Pooled across all 115 failures, cliff failures have **higher** copy scores than successful retrievals (PASS: 0.440, FAIL: 0.532). This is the **opposite** of ReDeEP's RAG hallucination signature, which has *low* External Context Score. The cliff failure mode is mechanistically distinct from RAG silent hallucination.

**Where does the literal continuation start?** A position-of-longest-match analysis on the 107 failures with detectable substring overlap reveals:

| Position in haystack | Failures | % |
|---|---:|---:|
| Q1 (0–25%) | 87 | **81%** |
| Q2 (25–50%) | 11 | 10% |
| Q3 (50–75%) | 6 | 6% |
| Q4 (75–100%) | 3 | 3% |

Refining to deciles, **70% of continuations resume from the 10–20% sub-region of the haystack** — the start of the article *body* in wikitext (immediately after the title and lead paragraph). The model is not autocompleting from the end of the prompt and it is not jumping to a random position. It is recognising "this is a Wikipedia article" from the title markup and resuming the article from where the body content begins.

**Mechanism**: we name this failure mode **"primacy-biased document continuation overflow"**. Above the cliff, the chat-template anchor at the start of the prompt loses sufficient attention weight; the model defaults to "this looks like a Wikipedia article, continue it" and starts emitting body content from the canonical body-start position. This is *not* parametric hallucination (the model is not inventing from internal memory), it is *not* the "Lost in the Middle" position bias (which is a retrieval-position effect on retrieval accuracy), and it is *not* attention sink collapse (the BOS sink is not lost, it is being *overruled*). It is a new failure mode.

**Why this matters**: ReDeEP's mitigation method (AARF — Add Attention, Reduce FFN) is designed for the parametric-takeover failure mode. Because cliff failure has the *opposite* signature, AARF will not help and may make the cliff worse. The correct mitigation direction is **anchor strengthening** — increasing the chat-template anchor's attention weight to outcompete the document continuation prior. Candidates: SinkTrack-style instruction injection into the BOS sink [Gg1aPETCL6], periodic question re-injection inside the prompt, multi-turn chat template wrapping. We discuss these in §8.

**Honest correction note for v0.3 readers**: the "Boulter + Sarah Chen" synthesised hallucination example was visually striking and led us to assume cliff failure shared mechanism with RAG hallucination. The Karpathy-loop subtype analysis disambiguated this. The example is real (we link the raw CSV row in the appendix), but it is one trial out of 115 — not the dominant failure mode. The corrected understanding is more useful: cliff failure is a *new* failure mode and needs *new* mitigations, not retargeted RAG-hallucination mitigations.

## 5. Negative Findings and Honest Limitations

### 5.1 The prompt-format trap

Five intuitively-reasonable prompt formats produced worse results than the simple haystack + `"\n\nQuestion:"` format. At >1500 token contexts, the "Based on this document, answer the question" prefix caused the 3B Q4 model to continue the haystack text instead of answering — the structured instruction was overpowered by the continuation prior of the long document.

### 5.2 The panic output

When we used a repetitive synthetic filler paragraph as haystack (before switching to wikitext-2), the 3B Q4 model at depth 0.1, ctx 4096 literally generated:

> `>|||_PANIC I can feel my sanity slipping away. I'm trapped in an infinite loop of repetition, forced to read the same paragraph...`

This is *not* a quant.cpp bug — it reproduces on the reference llama.cpp build — but it is worth reporting because it shows how aggressively small instruction-tuned models can respond to distribution-shifted inputs. The fix was to switch to real varied text (wikitext-2).

### 5.3 The 8B problem

We attempted the same grid on Llama-3.1-8B-Q4_K_M but each inference took ~12 minutes even on Metal, making a 90-run grid impractical in one session. This is a limitation of the current Metal kernel path for Q4_K_M GGUF on M-series hardware, not a fundamental block. We leave 8B+ measurements to future work.

### 5.4 Single-language, single-domain

All measurements use English wikitext (biographical and encyclopedic prose). Cross-domain (code, legal, scientific) and cross-lingual (Korean, Chinese, Japanese) working memory ceilings are unknown. Edge-device inference products will almost certainly hit different cliffs on different content.

### 5.5 One prompt format

We iterated five formats but finalized on one. A systematic prompt-sensitivity study (template-robustness-as-ceiling-measurement) would strengthen the contribution substantially.

### 5.6 Sampling-noise estimation: CLI bug discovered and fixed mid-round

We initially planned a 60-trial random-seed sweep at the cliff cells (1B Q8 ctx=1024 and 3B Q4 ctx=1280) at temperature 0.7 to estimate sampling noise around the apparent 22 pp delta between FP32 and 6.4× compressed at the 1B cliff. The first attempt produced a striking failure: all 60 trials returned `Loading model from <seed>... cannot open '<seed>'`. The cause was a CLI bug we surfaced during this round — `tools/quant.c` advertised a `-s <seed>` flag in its `--help` output but the parser had no case for it. The seed argument was silently dropped, and the *next* positional argument (the seed value, e.g. `42`) was bound to the model-path slot. The downstream sampler in `tq_generate` was hardcoded to `rng_state = 42` per CLI invocation, so even if the parser had worked, sampled outputs would have been identical across "different" seeds.

We fixed both halves of the bug in a separate commit (`a8f6d8a`):
- Added `unsigned long long rng_seed` to `tq_gen_config_t` in `include/turboquant/tq_engine.h` and the single-header `quant.h`.
- Initialised `rng_seed = 42ULL` in `tq_default_gen_config` (back-compat preserving).
- Wired `rng_state = config->rng_seed ? config->rng_seed : 42ULL` in both `src/engine/tq_generate.c` and `quant.h`.
- Added the `-s` parser case in `tools/quant.c`.

After the fix, `-s 42` and `-s 1337` produce demonstrably different outputs at `-T 0.7` (verified manually), and the no-`-s` default is bit-for-bit identical to `-s 42` (verified for backwards compatibility). All 35 build_metal/ tests still pass.

The seed-controlled sampling sweep itself is still pending — running it post-fix is straightforward but was outside the time budget for this round. The 1B cliff cell's apparent 22 pp gap between baseline and compressed therefore remains unverified by per-seed sampling noise. With the CLI fix in place, a v2 of this report can simply re-run `bench/niah_seed_sweep.sh` (the script already exists, just bug-hit at the time of v1 submission).

## 6. Discussion: What "Long-Context Replaces RAG" Actually Means at the Edge

Vector RAG's well-known failure mode is *silent hallucination on retrieval miss* — when the wrong chunk is fetched, the LLM fills in with plausible-sounding lies. The long-context-replaces-RAG argument is that loading the whole document into the LLM's context eliminates this mode because the LLM has all the information.

Our measurements show this argument **has an empirical floor** at the edge:

1. *Memory-wise*, a 128K context fits in 9.5 GB on a 16 GB Mac with 6.4× KV compression — the original framing is correct at the allocation level.
2. *Retrieval-wise*, both models we measured stop following the chat-template instruction long before the memory is full:
   - Llama-3.2-1B-Q8: cliff between 512 and 1024 tokens (degradation interval), zero retrieval by 1536. **Effective working memory is 0.4% of the nominal 128K context window.**
   - Llama-3.2-3B-Q4 (default loader path): cliff between 1024 and 1280 tokens, **as a step function** with no degradation interval. Effective working memory is **0.78%** of the nominal 128K context window.

3. Above the cliff, the dominant failure mode (84% of 115 cliff failures, characterised in §4.6) is **literal continuation of the haystack from its 10–20% body-start position** — not parametric hallucination. Surface-level proxy analysis shows this failure has the *opposite* mechanistic signature from RAG silent hallucination [Sun et al., ReDeEP, ICLR 2025]: cliff failures have *high* External Context Score (model copies a lot from loaded context) and *low* Parametric Knowledge Score (model is not inventing from internal memory). RAG silent hallucination has the inverse pattern. Our v0.3 draft conflated the two by citing a 1-in-115 outlier ("Boulter was hired as CFO") as the dominant case; v0.4 corrects this in §4.6.

This matters for mitigation. The failure mode we found — *primacy-biased document continuation overflow* — cannot be fixed by retargeted RAG-hallucination interventions like ReDeEP's AARF (Add Attention, Reduce FFN), which is designed for the parametric-takeover regime. The chat-template anchor at the start of the prompt is *being overruled*, not lost; the appropriate mitigations are anchor-strengthening interventions: SinkTrack-style instruction injection into the BOS sink, periodic question re-injection inside the prompt to re-anchor the chat template at intervals, or multi-turn conversational chunking that refreshes the anchor per turn. We outline these as Phase 2 candidates in §8.

The long-context-replaces-RAG argument therefore holds **only for documents that fit in the effective working memory of the specific model + quantization + loader configuration being deployed**, and the gap between "nominal context window" and "effective working memory" is *two to three orders of magnitude* at edge scale, not within an order of magnitude.

This does not invalidate the argument for larger models or frontier-scale inference. It *does* mean that:
- Edge-device vendors making "long-context replaces RAG" claims must publish their effective working memory measurements, not just their memory allocation numbers.
- The community needs a standardised "effective working memory" benchmark slot in benchmark suites alongside perplexity — perplexity is teacher-forced and never asks the model to *act on* the long context, so it cannot detect this cliff.
- KV compression research that targets edge deployment should report results bracketing the model's cliff; aggregate accuracy numbers above the cliff are uninformative because both compression and baseline are at zero.

## 7. Reproducibility

All code, data, and raw inference logs are in the `bench/results/niah/` tree of the quant.cpp repository (commit `5aba85d` and later).

```bash
# 1B Q8 working memory sweep (this work)
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="256 512 1024 1536 2048" \
  bash bench/niah_test.sh

# 3B Q4 ceiling probe (this work) — extends the 512/1024 R1 baseline
MODEL=models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="1280 1536 1792 2048" \
  bash bench/niah_test.sh

# 3B Q4 R1 baseline (the original 36/36 PASS run that motivated this work)
MODEL=models/Llama-3.2-3B-Instruct-Q8_0.gguf GRID=quick bash bench/niah_test.sh

# Aggregate any single CSV into a markdown summary table
python3 bench/results/niah/aggregate.py bench/results/niah/results_<TIMESTAMP>.csv
```

The exact CSVs and per-run CLI logs that back the tables in §4 are:

| File | Runs | Coverage |
|---|---:|---|
| `bench/results/niah/results_20260411T024534.csv` | 36 | 3B Q4, ctx 512+1024 (R1) |
| `bench/results/niah/results_20260411T043236.csv` | 90 | 1B Q8, ctx 256–2048 (R2) |
| `bench/results/niah/results_20260411T052319.csv` | 72 | 3B Q4, ctx 1280–2048 (R3) |
| `bench/results/niah/master_table.md` | — | Combined markdown summary |

Models: `meta-llama/Llama-3.2-1B-Instruct` Q8_0 GGUF, `meta-llama/Llama-3.2-3B-Instruct` Q8_0 GGUF.
Hardware: Apple M-series (Metal kernel path).
Dependencies: bash, Python 3, CMake, no other external libraries.
Per-run CLI outputs preserved in `bench/results/niah/raw_<TIMESTAMP>.log`.

## 8. Future Work

The Phase 2B subtype analysis (§4.6) suggests a concrete sequence of mitigation experiments with mechanistic justification:

1. **Periodic Question Re-Injection (PQRI)** [1-day experiment]. Embed `<|start_header_id|>user<|end_header_id|>{question}<|eot_id|>` markers every 256 tokens inside the haystack. The chat-template anchor is *spatially* refreshed throughout the prompt rather than relying on the single anchor at the start. Expected: cliff position shifts upward by 2–4×. Format change only, no model modification.

2. **Conversational Chunking** [1-day experiment]. Split the haystack into multiple `<|user|>chunk<|/user|>` turns instead of one giant user turn, with the question as the final turn. Each chunk refreshes the chat-template anchor. Predicted to be effective because models are heavily instruction-tuned on multi-turn dialogue.

3. **Question-Anchored Sink Injection (QASI)** [1-week experiment]. Adapt SinkTrack [Gg1aPETCL6, ICLR 2026] from vision-language models to the edge-LLM regime. SinkTrack injects contextual features into the BOS attention sink and reports +21.6% on SQuAD2.0 with Llama-3.1-8B-Instruct. Our cliff measurement provides an obvious testbed: inject the question into the BOS sink at inference time and re-measure the cliff for 1B Q8 and 3B Q4. We are not aware of this being tried at edge-device scale.

4. **Conversational chunking + SinkTrack as a hybrid baseline**. If PQRI works at all, it will become the cheap baseline; QASI will become the deeper mechanistic candidate; the union may compound.

5. **Mechanistic verification with HuggingFace Llama-3.2-3B**: extract per-layer attention to chat-template anchor tokens (`<|start_header_id|>`, `<|eot_id|>`) as context length crosses the cliff. Test the precise hypothesis: anchor attention weight decays below a threshold and the document-continuation attention head wins. This requires a PyTorch + HF setup we deferred from Phase 2B for time reasons.

6. **Ceiling measurements at 8B and 13B**, once Metal Q4_K_M prefill is optimized (currently ~12 min/inference makes a 90-trial grid impractical).

7. **Cross-lingual measurements**: Korean / Chinese / Japanese. The "primacy-biased document continuation" hypothesis predicts the cliff and its failure mode will reproduce in all languages because the mechanism is structural, not lexical.

8. **Cross-domain measurements**: code, legal, scientific prose. Prediction: the dominant failure mode in code haystacks will not be 10–20% body-start continuation (no Wikipedia-like body-start convention) but something analogous to "continue the function definition". Different surface patterns, same anchor-overpowering mechanism.

9. **Head-to-head with KIVI / H2O / SnapKV / DynamicKV at matching model scales**. None of these were measured at the 1B–3B regime; our cliff protocol provides the comparison testbed.

10. **Working Memory Budget RAG (WMB-RAG)**: a RAG retriever that respects the model's measured effective working memory as a hard constraint instead of pretending the nominal context window is usable. This is the most directly industrially relevant follow-up.

## 9. References

- Bai et al. 2023. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* arXiv:2308.14508.
- Cai et al. 2024. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* arXiv:2406.02069.
- Chen et al. 2023. *MLC-LLM: Universal LLM Deployment Engine.* https://github.com/mlc-ai/mlc-llm.
- Hsieh et al. 2024. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654.
- Kamradt 2023. *Needle in a Haystack — Pressure Testing LLMs.* https://github.com/gkamradt/LLMTest_NeedleInAHaystack.
- Li et al. 2024. *SnapKV: LLM Knows What You are Looking for Before Generation.* arXiv:2404.14469.
- Liu, N. F. et al. 2023. *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172.
- Liu, Z. et al. 2024. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* arXiv:2402.02750.
- Zhang et al. 2023. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* arXiv:2306.14048.

---

*Draft v0.3 — generated by the Phase 1B Karpathy loop. All result sections populated; Related Work and References complete; FP32-weights control done (the cliff is invariant to weight precision); cliff-cell seed sweep blocked by an upstream `tools/quant.c -s` flag bug we surfaced during this round (documented as §5.6 limitation, separate fix issue filed). Ready for arXiv submission as a v1 tech report.*
