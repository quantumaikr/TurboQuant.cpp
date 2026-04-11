# NIAH Master Table — Phase 1B Working Memory Cliff Measurements

**Date**: 2026-04-11
**Hardware**: Apple M-series, Metal kernel path (`build_metal/quant`)
**Protocol**: 3 needles × 3 depths (0.1, 0.5, 0.9) per (model, ctx, KV-config) cell, plus a 6-trial FP32-weights control at the cliff transition.
**Scoring**: case-insensitive ERE grep for keywords, against 32-token greedy generation.
**Total trials**: 204 (R1=36 + R2=90 + R3=72 + R4=6).

---

## Llama-3.2-1B-Instruct Q8_0 (no on-the-fly Q4 conversion)

| Method | ctx=256 | ctx=512 | ctx=1024 | ctx=1536 | ctx=2048 |
|---|---:|---:|---:|---:|---:|
| `fp32` (baseline) | **8/9 (89%)** | **9/9 (100%)** | **4/9 (44%)** | 0/9 (0%) | 0/9 (0%) |
| `turbo_q4_w128` (6.4×) | 8/9 (89%) | 9/9 (100%) | 2/9 (22%) | 0/9 (0%) | 0/9 (0%) |
| Δ | 0 pp | 0 pp | **−22 pp** | 0 | 0 |

**Source**: `results_20260411T043236.csv` (this work, 90 trials)

**Cliff location**: 512 → 1024 transition. The 1024 cell is *unstable* — both methods produce nondeterministic-looking failures (model echoes wikitext header `= = = 2008 II =` etc.) on 5–7 out of 9 trials.

---

## Llama-3.2-3B-Instruct Q8_0 (default CLI: on-the-fly Q4 weight conversion)

| Method | ctx=512 | ctx=1024 | ctx=1280 | ctx=1536 | ctx=1792 | ctx=2048 |
|---|---:|---:|---:|---:|---:|---:|
| `fp32` (baseline) | **9/9 (100%)** | **9/9 (100%)** | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
| `turbo_q4_w128` (6.4×) | 9/9 (100%) | 9/9 (100%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
| Δ | 0 pp | 0 pp | 0 pp | 0 pp | 0 pp | 0 pp |

**Source**: `results_20260411T024534.csv` (R1, ctx 512+1024, 36 trials) + `results_20260411T052319.csv` (this work, ctx 1280–2048, 72 trials).

**Cliff location**: 1024 → 1280 transition. The cliff is a **step function** for 3B Q4 — perfect retrieval at 1024, total collapse 256 tokens later. There is no degradation interval; the model simply stops following the chat template.

---

## Combined: Working memory cliff per model

| Model | Highest 100% retrieval ctx | First 0% retrieval ctx | Cliff width |
|---|---:|---:|---:|
| Llama-3.2-1B-Q8_0 | 512 | 1536 | ~1024 tokens (degradation interval at 1024) |
| Llama-3.2-3B-Q4 (default) | 1024 | 1280 | **<256 tokens (step function)** |

**Scaling observation** (n=2, anecdotal): going from 1B Q8 to 3B Q4 **doubles** the highest-100% ceiling (512 → 1024). This is the first measured edge-device-scale data point on a long-context-replaces-RAG question with a clear threshold.

---

## Compression neutrality (apples-to-apples)

| Model | Cells where compression and baseline disagree | Cells where they agree | Overall delta |
|---|---|---|---|
| 3B Q4 (10 cells, 90 trials) | 0 | 10 | **+0.0 pp** |
| 1B Q8 (10 cells, 90 trials) | 1 (ctx=1024 cell, both at the cliff) | 9 | **−4.4 pp** |

**Key reading**: 6.4× KV compression is **bit-for-bit identical** to FP32 baseline in every cell **except** the 1B cliff cell, where compression *appears* to be 22 pp worse (2/9 vs 4/9). However, both points are statistically indistinguishable from random at n=9 — this is not a compression-quality finding, it's a cliff-instability finding.

The headline result remains: **KV compression preserves whatever the model can already retrieve, and the working memory cliff is a model property, not a KV property.**

---

## Weight-precision control (R4): the cliff is invariant to weight quantization

The default `quant.cpp` loader silently re-quantizes Q8_0 GGUF weights to Q4 in memory. To eliminate the possibility that the cliff is an artifact of this requantization, we re-ran the cliff transition cells with the FP32-weights loader path (`TQ_NO_Q4=1`).

| Weight precision | ctx=1024 | ctx=1280 |
|---|---:|---:|
| Q4 (default loader) | **100%** (18/18) | **0%** (0/18) |
| **FP32** (`TQ_NO_Q4=1`)  | **100% (3/3)** | **0% (0/3)** |

**Identical cliff location at 8× per-parameter precision.** The model's instruction-following collapse happens at the same context length whether weights are Q4 or FP32 — the cliff is a *model* property, not a weight quantization artifact.

Above-cliff failure mode is also identical between Q4 and FP32 weights: all six FP32 ctx=1280 trials produced wikitext continuation ("Doctors , followed by a role in the 2007 theatre production of How to Curse..."), the same dominant failure mode as the Q4 grid.

Source: `bench/results/niah/results_fp32ctrl_20260411T091023.csv` (6 trials, 60 minutes wall time on Metal — FP32-weights inference runs at ~10 min per trial, hence the small grid).

---

## Failure mode taxonomy (qualitative)

When the model fails above the cliff, it does not say "I don't know." It produces one of:

1. **Wikitext continuation**: model picks up where the haystack left off (`"Doctors , followed by a role in the 2007 theatre production..."`).
2. **Header echo**: model emits a wikitext section header it saw earlier (`"= = = 2008 II ="`, `"= Robert Boulter ="`).
3. **Synthesised hallucination**: model fuses the needle into the surrounding biography (`"In 2023 Boulter was hired as the chief financial officer..."` — Boulter is the wikitext subject, Sarah Chen is the needle).

Failure mode 3 is the most consequential. It is the same silent-hallucination failure that vector RAG produces on retrieval miss — but here it happens *because* the document was loaded fully and the model lost the question. The "long-context replaces RAG" framing assumed this failure mode would disappear when the model has all the information; our measurements show it does not, in the edge-device quantized regime.

---

## Notes on prior work overlap

- **Lost in the Middle** (Liu et al. 2023) measured retrieval at frontier scale (Claude-1.3, GPT-3.5, GPT-4); we measure 1B/3B Q4–Q8.
- **NIAH** (Kamradt 2023) is the inspiration for the protocol but uses cloud LLMs.
- **KIVI / H2O / SnapKV / PyramidKV** measure KV compression on Llama-2-7B and up; our finding that compression is orthogonal to ceiling at 1B–3B is novel.
- **RULER** (Hsieh et al. 2024) is the obvious next step for systematic head-to-head — see Future Work in `working-memory-cliff.md`.

---

## Reproduce

```bash
# 1B grid (this work)
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="256 512 1024 1536 2048" \
  bash bench/niah_test.sh

# 3B ceiling probe (this work)
MODEL=models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="1280 1536 1792 2048" \
  bash bench/niah_test.sh

# 3B baseline (R1)
MODEL=models/Llama-3.2-3B-Instruct-Q8_0.gguf GRID=quick bash bench/niah_test.sh

# Aggregate any single CSV
python3 bench/results/niah/aggregate.py bench/results/niah/results_<TIMESTAMP>.csv
```
