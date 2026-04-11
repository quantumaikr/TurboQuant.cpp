# Twitter/X launch thread — Working Memory Cliff

*~10 tweets, ~280 chars each. Designed to drive readers to the HF blog post and the arXiv-style tech report on GitHub.*

---

**1/**
We measured how much of a long context two popular edge-device LLMs (Llama-3.2-1B-Q8, Llama-3.2-3B-Q4) can actually use.

Both fall off a cliff at less than **1% of their advertised 128K context window**.

This is the empirical reality check the "Beyond RAG" argument needed. 🧵

---

**2/**
The 3B cliff is a **step function**:

| ctx | retrieval |
|---|---|
| 1024 | 100% (18/18) |
| 1280 | **0% (0/18)** |

256 tokens of context length, perfect → broken. No degradation interval. The model just stops following the instruction.

---

**3/**
The 1B cliff is graded but starts much earlier:

| ctx | fp32 KV | 6.4× compressed |
|---|---|---|
| 512 | 100% | 100% |
| 1024 | 44% | 22% |
| 1536 | 0% | 0% |

Same wikitext-2 haystack, same three needles, n=9 per cell.

---

**4/**
Important: KV cache compression (6.4×) is **bit-for-bit identical** to FP32 baseline in 18 of 20 cells.

The single disagreement is at the cliff cell where both methods are statistical noise.

**The cliff is a model property, not a KV property.**

---

**5/**
Above the cliff, the failure mode is *not* "I don't know."

Three modes we observed:
1. Wikitext continuation (model picks up the haystack)
2. Header echoes ("= = = 2008 II =")
3. Synthesised hallucinations — and #3 is the consequential one.

---

**6/**
Example synthesised hallucination, at 1B fp32 ctx=1024:

Haystack: Wikipedia article about Robert Boulter (English actor).
Needle: "The CFO of Northwind Logistics is Sarah Chen, hired in 2023."

Model output: *"In 2023 Boulter was hired as the chief financial officer..."*

It **fused** them.

---

**7/**
This is the same silent-hallucination failure mode that vector RAG produces on retrieval miss — happening in the regime that was supposed to *eliminate* it.

"Load the whole document, skip the chunker, hallucination goes away." Works at frontier scale. Doesn't work at 1B–3B Q4.

---

**8/**
Why this matters for "Beyond RAG":

- *Memory-wise*, 128K context fits in 9.5GB on a 16GB Mac with 6.4× KV compression. ✅
- *Retrieval-wise*, the 3B Q4 model stops following instructions at ~1024 tokens, regardless of compression. ❌

Two to three orders of magnitude gap.

---

**9/**
Honest scope: 2 models, 3 needles, 1 language (English), 1 domain (Wikipedia bio). 198 NIAH trials total.

The cliff might:
- Sit higher for 8B+ models
- Move with prompt format
- Be reducible with long-context fine-tuning

If you can falsify it, we want the data.

---

**10/**
Full tech report (arXiv-style):
github.com/quantumaikr/quant.cpp/blob/main/docs/paper/working-memory-cliff.md

HF blog post:
[link when published]

Reproduce in 5 minutes:
github.com/quantumaikr/quant.cpp/blob/main/bench/niah_test.sh

Pull requests welcome. /end

---

## Posting strategy

- Post #1 alone, wait 1-2 min for early signal
- Then thread the rest as a single drop
- Pin the thread on the project account
- Cross-post the lead tweet to:
  - r/LocalLLaMA (with the title: *"Measuring the working memory cliff of Llama-3.2-1B/3B: 198 NIAH trials, KV compression is bit-for-bit identical to baseline, both models fall off a cliff at <1% of nominal context"*)
  - HackerNews (Show HN: title)
- Tag in HF post: @huggingface @abacaj @karpathy (only if they engage with edge-LLM benchmarking content; do not spam-tag)

## Anticipated responses + drafts

**Q: "n=9 is too small."**
A: We agree. Cliff-cell seed sweep with 5 seeds × 3 needles = 15/cell is in the tech report appendix. The headline cells have wider error bars; the cliff direction does not change.

**Q: "Why not run 8B?"**
A: Tried. Q4_K_M GGUF on Metal runs at ~12 min per inference for 3B, not feasible for a 90-trial grid. We're working on the Metal Q4_K_M kernel; expect 8B numbers in v2.

**Q: "Is this a quant.cpp bug? Does llama.cpp do the same?"**
A: We tested with the exact same Q8_0 GGUF. The cliff reproduces under both quant.cpp's default loader and the FP32-weights path (`TQ_NO_Q4=1`). It is not specific to a quantization scheme.

**Q: "Did you try a system prompt?"**
A: We tried six prompt formats including system prompts, structured `<<<DOCUMENT>>>` delimiters, few-shot Q/A, and raw continuation. The simple format we report was the best of the six. Format-sensitivity is documented in §5.1 of the report.

**Q: "RULER / LongBench would be better."**
A: Yes. RULER is the obvious next step and is in §8 (Future Work). This is a v1 measurement explicitly scoped to the question "where does the cliff sit at edge scale" — not a comprehensive long-context benchmark.
