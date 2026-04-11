# The Working Memory Cliff: Why Long-Context Inference Doesn't Replace RAG (Yet) at the Edge

*A reality check on the "Beyond RAG" argument for 1B–3B quantized LLMs*

---

## TL;DR

We measured how much of a long context two popular edge-device LLMs can *actually* use, and found a sharp cliff:

- **Llama-3.2-1B-Q8** retains 100% retrieval at **512 tokens**, drops to 0% by **1536** — across the cliff in ~1024 tokens.
- **Llama-3.2-3B-Q4** retains 100% at **1024 tokens**, drops to 0% at **1280** — across the cliff in **<256 tokens, as a step function**.

Both models reach effective working memory at **less than 1% of their nominal 128K context window**.

We also measured **6.4× KV cache compression** vs FP32 baseline across the same grid. The result: 18 of 20 cells are bit-for-bit identical between baseline and compressed. **The cliff is a model property, not a KV property.**

This is the empirical reality check the "long-context inference replaces RAG" argument needed for the edge-device case.

[**📄 Full tech report (arXiv-style)**](https://github.com/quantumaikr/quant.cpp/blob/main/docs/paper/working-memory-cliff.md) · [**📊 Raw data + reproduction scripts**](https://github.com/quantumaikr/quant.cpp/tree/main/bench/results/niah)

---

## Why this matters

Here's the popular argument:

> "Vector RAG hallucinates when the wrong chunk is retrieved. Modern LLMs have 128K+ context windows. Just load the whole document, skip the chunker, the silent-hallucination failure goes away."

It's a clean argument. It is also true at frontier scale (Gemini 1.5 Pro, Claude, GPT-4 Turbo). And it is correct **at the memory-allocation level** even on a 16 GB Mac: with 6.4× KV cache compression, a 128K context for Llama-3.2-3B fits in about 9.5 GB.

But there is a measurable gap between *can the cache hold the document* and *can the model still follow your question after loading it*. We wanted to know how big that gap is for the small quantized models that ship with `pip install quantcpp`, `ollama pull`, and `llama.cpp` — the models real users actually run.

So we ran the simplest possible experiment: insert a fact into a longer document, ask a question whose answer is the fact, see if the model retrieves it. Vary the document length. Vary the depth. Vary the needle. Measure 198 trials.

## The cliff

**Llama-3.2-1B-Instruct-Q8_0** (no on-the-fly Q4 conversion in the loader):

| Context | fp32 KV | 6.4× compressed KV |
|---:|:-:|:-:|
| 256 | 89% | 89% |
| **512** | **100%** | **100%** |
| 1024 | 44% | 22% |
| 1536 | 0% | 0% |
| 2048 | 0% | 0% |

**Llama-3.2-3B-Instruct-Q4** (default `quant.cpp` loader, on-the-fly weight requantization):

| Context | fp32 KV | 6.4× compressed KV |
|---:|:-:|:-:|
| 512 | 100% | 100% |
| **1024** | **100%** | **100%** |
| **1280** | **0%** | **0%** |
| 1536 | 0% | 0% |
| 1792 | 0% | 0% |
| 2048 | 0% | 0% |

The 3B cliff is **a step function**. Perfect at 1024, total collapse 256 tokens later. There's no degradation interval. The model goes from following the chat-template instruction perfectly to ignoring it completely within a 25% context-length step.

## The failure mode is *not* "I don't know"

Above the cliff, the model doesn't say "I don't know" or "the document doesn't mention it." It does one of three things:

**1. Wikitext continuation** — picks up where the haystack left off:
> *"...Doctors , followed by a role in How to Curse directed by Josie ..."*

**2. Section header echo** — emits a wikitext header it saw earlier in context:
> *"= = = 2008 II ="*

**3. Synthesised hallucination** — and this is the one that should worry production RAG users:
> *"In 2023 Boulter was hired as the chief financial officer..."*

The haystack is a Wikipedia article about Robert Boulter, an English actor. The needle says "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023." The model **fused them**: it produced a coherent invented sentence that grafts the needle's "2023 hire" onto Boulter's biography.

This is the **same silent-hallucination failure mode that vector RAG produces on retrieval miss** — happening in the regime that was supposed to *eliminate* it.

## KV compression is orthogonal to the cliff

We also wanted to know if our 6.4× KV cache compression made the cliff worse. The short answer is *no*: 18 of 20 (model × ctx × method) cells are identical between baseline and compressed.

| Model | Disagreeing cells | Agreeing cells | Overall delta |
|---|---|---|---|
| 3B Q4 (10 cells, 90 trials) | 0 / 10 | 10 / 10 | **+0.0 pp** |
| 1B Q8 (10 cells, 90 trials) | 1 / 10 (cliff cell, both at noise) | 9 / 10 | −4.4 pp (within noise) |

The single disagreement is at the 1B cliff cell where both methods are statistically indistinguishable from random anyway. Outside the cliff, every cell is exact.

This matters because it means **KV compression research that targets edge deployment should report results that bracket the cliff, not aggregate across it**. Above the cliff both compression and baseline are at zero — uninformative. Below the cliff both are at 100% — uninformative. The interesting comparison is the *graded* region in between, which exists for 1B but not for 3B Q4.

## What this changes about "Beyond RAG"

Our prior post (["Beyond RAG"](https://github.com/quantumaikr/quant.cpp/blob/main/docs/beyond-rag-manifesto.md)) showed that loading a full 600-token corporate document into a 3B model with 6.4× KV compression got 7/7 answers correct, while a chunk-RAG baseline got 0/7 with silent hallucinations. That result is real and reproducible. But it was measured on a *short* document — well below the cliff we now know exists.

The honest reframing is:

> **Beyond RAG works when the document fits in the model's effective working memory. For Llama-3.2-3B-Q4 on the default loader path, that is approximately 1024 tokens — not 128K.**

Two to three orders of magnitude smaller than the nominal context window. Edge-device vendors who claim "long context replaces RAG" should publish their effective working memory measurements alongside their memory allocation numbers, because the gap is *not* within an order of magnitude — it's enormous.

## What we want to be wrong about

We measured two models, three needles, single language (English), single content domain (Wikipedia biography). The cliff might:

- Sit higher for larger models (8B, 13B, 70B).
- Sit lower for non-English text or out-of-distribution domains.
- Move with prompt format (we tried 6 variants and finalized on the most permissive).
- Be reducible with instruction-tuning targeting long-context retrieval.
- Be moveable with `--repeat-penalty`, `--temperature`, or `--top-p` settings we didn't probe.

If you can falsify any of this, we want the data. The benchmark is one bash script and 200 lines of Python — pull request welcome.

## Reproduce in 5 minutes

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build_metal -DTQ_BUILD_METAL=ON && cmake --build build_metal -j8

# Download models (Q8_0 GGUF from HuggingFace)
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir models
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q8_0.gguf --local-dir models

# 1B working memory sweep (~30 min)
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="256 512 1024 1536 2048" \
  bash bench/niah_test.sh

# 3B ceiling probe (~60 min)
MODEL=models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="1280 1536 1792 2048" \
  bash bench/niah_test.sh

# Aggregate
python3 bench/results/niah/aggregate.py bench/results/niah/results_<TIMESTAMP>.csv
```

Models, raw CSV, per-run CLI logs, and the markdown master table are all under `bench/results/niah/` in the repo. The full tech report is at [`docs/paper/working-memory-cliff.md`](https://github.com/quantumaikr/quant.cpp/blob/main/docs/paper/working-memory-cliff.md).

## Takeaways

1. **Effective working memory ≪ nominal context window** for edge-device quantized LLMs. Less than 1% of the advertised 128K, in the two models we measured.
2. **The cliff is a model property**, not a KV cache property. 6.4× compression preserves whatever the model can retrieve.
3. **Above the cliff, the failure mode is silent hallucination** — the same failure mode that "long context replaces RAG" was supposed to eliminate.
4. **"Beyond RAG" still works** — but only for documents that fit in the model's effective working memory, which is much smaller than the cache allocation suggests.

For us, the practical implication is: keep measuring, publish the cliff alongside the compression ratio, and stop framing memory-fits-in-cache as the same thing as model-can-use-it.

---

*Raw data, reproduction scripts, and the full tech report (arXiv-style) are in [quant.cpp on GitHub](https://github.com/quantumaikr/quant.cpp). If you reproduce this on a different model and find a different cliff, [open an issue](https://github.com/quantumaikr/quant.cpp/issues) — we want to be wrong about something.*
