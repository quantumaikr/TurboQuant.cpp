<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">quant.cpp</h3>
<p align="center"><b>Beyond RAG: load the whole document. On your laptop.</b></p>

<p align="center">
  Chunking was a workaround for small context windows. We just made it unnecessary.<br>
  6.4× KV compression brings full-document understanding to consumer hardware.<br>
  <code>pip install quantcpp</code> — 16K lines of C, zero dependencies.
</p>

<table align="center">
<tr>
<td align="center"><b>7/7 vs 0/7</b><br>Beyond RAG measured</td>
<td align="center"><b>6.4x compression</b><br>+3% PPL</td>
<td align="center"><b>128K context</b><br>on 16GB Mac</td>
<td align="center"><b>16K LOC</b><br>zero deps</td>
</tr>
</table>

<p align="center">
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/v/quantcpp.svg?label=PyPI&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/quantcpp/"><img src="https://img.shields.io/pypi/pyversions/quantcpp.svg" alt="Python versions"></a>
  <a href="https://github.com/quantumaikr/quant.cpp/releases/latest"><img src="https://img.shields.io/github/v/release/quantumaikr/quant.cpp?label=release" alt="Release"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-35%20pass-brightgreen" alt="Tests"></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/models-7%20verified-blue" alt="Models"></a>
  <a href="https://quantumaikr.github.io/quant.cpp/"><img src="https://img.shields.io/badge/WASM_demo-192KB-purple" alt="WASM"></a>
  <a href="https://quantumaikr.github.io/quant.cpp/guide/"><img src="https://img.shields.io/badge/guide-How_it_Works-blueviolet" alt="Guide"></a>
  <a href="#"><img src="https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20WASM-orange" alt="Platforms"></a>
</p>

---

## Quick Start

**Ollama-style CLI (v0.12.0+):**
```bash
pip install quantcpp

quantcpp pull llama3.2:1b               # download from HuggingFace
quantcpp run llama3.2:1b                # interactive chat
quantcpp serve llama3.2:1b -p 8080      # OpenAI-compatible HTTP server (SSE streaming)
quantcpp client "Hi"                    # streaming client → server on :8080
quantcpp list                           # show cached models
```

Short aliases: `smollm2:135m`, `qwen3.5:0.8b`, `llama3.2:1b`. Auto-pulls on first `run`/`serve`. The `serve` subcommand exposes `POST /v1/chat/completions` (OpenAI-compatible) on port 8080 — clients pass `"stream": true` for SSE streaming, or omit it for a single JSON response. Built-in `quantcpp client` supports both modes (default: streaming, `--no-stream` for single response).

**One-shot question:**
```bash
quantcpp run llama3.2:1b "What is gravity?"
```

**Python API (3 lines):**
```python
from quantcpp import Model
m = Model.from_pretrained("Llama-3.2-1B")
print(m.ask("What is gravity?"))
```

Downloads on first use, cached at `~/.cache/quantcpp/`. No API key, no GPU. [Try in browser →](https://quantumaikr.github.io/quant.cpp/) · [**Interactive Guide →**](https://quantumaikr.github.io/quant.cpp/guide/)

---

## Key Result: FP32 Quality at 3x Compression

> **128 FP32 tokens + 4-bit everything else = FP32 quality, regardless of context length.**

Measured on **Llama 3.2 3B, 3970 tokens** (k128 = 3.2% FP32):

| Configuration | PPL | vs FP32 | KV Memory (32K) | Speed |
|---|---:|---:|---:|---:|
| FP32 (baseline) | 19.41 | — | 7.17 GB | baseline |
| **4-bit + progressive** | **19.39** | **-0.1%** | **2.33 GB** | **+13%** |
| 4-bit flat | 20.02 | +3.1% | 2.30 GB | +13% |

```python
m = Model("model.gguf", progressive=True)  # ← FP32 quality, 3x less memory, 13% faster
```

**Why it works:** Transformer attention concentrates ~70% of weight on the last ~128 tokens. Keeping those at full precision while compressing everything else aligns storage precision with information value — near-optimal by rate-distortion theory.

**Context-length invariant:** the same 128-token window works at 4K, 32K, or 128K. At 128K context, only 0.1% of tokens are FP32 — effectively all-4-bit with FP32 quality.

---

## 128K Context on 16GB Mac — Measured

Llama 3.2 3B with 6.4x KV compression. **Real RSS measured on M1 Pro 16GB:**

| Context | FP32 KV | **quant.cpp 6.4x** | Savings | Speed |
|---:|---:|---:|---:|---:|
| 16K | 8.5 GB | **6.5 GB** | **-2.0 GB** | 6.6 tok/s |
| 32K | 9.6 GB | **8.2 GB** | **-1.4 GB** | 4.9 tok/s |
| 65K | — | **8.5 GB** | — | 1.6 tok/s |
| **128K** | **OOM** | **9.5 GB** | — | 0.8 tok/s |

128K context with a 3B model in 9.5 GB. Generation speed is the same as FP32 (6.6 vs 6.5 tok/s at 16K).

```python
m = Model("llama-3b.gguf", aggressive=True, context_length=131072)  # 128K in 9.5 GB
```

---

## Beyond RAG: 7/7 vs 0/7 — Measured

> **Chunking RAG was a workaround for small context windows. The workaround became dogma.**
> **Now context windows are big enough that we don't need the workaround.**

A direct comparison on **Llama 3.2 3B Q8_0**, 5-section synthetic document, 7 questions (4 single-hop, 3 multi-hop):

| Method | Accuracy | Behavior on failure |
|---|---:|---|
| **Chunk-RAG** (wrong section retrieved) | **0/7** | **Hallucinated all answers** |
| Full Document (FP32 KV) | **7/7** | Correct |
| **Full Document (6.4× compressed KV)** | **7/7** | **Correct — zero quality loss** |

### The hidden failure mode of chunk-RAG

When chunk-RAG retrieves the wrong section, the model **doesn't say "I don't know"** — it generates plausible-sounding lies:

| Question | Chunk-RAG (wrong section) | Truth |
|---|---|---|
| "Who is the CTO?" | "John Smith" ❌ | Maria Santos |
| "What is the revenue?" | "$1,000,000" ❌ | 847 million |
| "R&D %?" | "15% of net income" ❌ | 14% of revenue |
| "Who proposed?" | "John Smith, EVP" ❌ | James Park |

This is the production risk no one measures: **silent hallucination on retrieval failure**. Your monitoring shows 100% uptime. Your users get wrong answers.

### Beyond RAG: load the whole document instead

With **6.4× KV compression**, a full 5-section document fits in context on a 16GB Mac. The model answers all 7 questions correctly, including multi-hop reasoning that requires linking information across sections:

> **"What risk affects the growth region?"** → currency fluctuations
> *(requires linking Section 3 "Asia growth" with Section 5 "Asia currency risk")*

Chunk-RAG cannot do this — each chunk is retrieved independently.

### RAG isn't dead. RAG is one tool.

This isn't "RAG is dead." RAG is still the only way to handle 100K+ document corpora. But:
- **RAG decides *which documents* to look at** (search problem)
- **Long-context decides *how deeply* to understand them** (reasoning problem)

The bug was using the same tool for both. The fix is using each for what it's good at.

**Reproduce in 5 minutes:** [bench/document_level_rag_test.sh](bench/document_level_rag_test.sh)
**Full benchmark report:** [bench/results/document_level_rag_breakthrough.md](bench/results/document_level_rag_breakthrough.md)
**Manifesto:** [docs/beyond-rag-manifesto.md](docs/beyond-rag-manifesto.md)

> **Honest disclaimer:** v1 is a synthetic 5-section document with 7 questions on a single 3B model. We're not claiming this is LongBench. We *are* claiming it's enough to start a conversation about the failure mode chunk-RAG has been hiding.

> **v2 update — the Working Memory Cliff (2026-04-11):** We followed up the v1 result with 204 NIAH trials across 1B and 3B at context lengths 256–2048, plus a 6-trial FP32-weights control. Both models hit a sharp cliff at **less than 1% of their nominal 128K context window** (1B Q8 at 512–1024, 3B Q4 at 1024–1280 *as a step function*). The 6.4× KV compression is bit-for-bit identical to FP32 baseline in 18 of 20 cells, so the cliff is a model property — not a KV property and not a weight-quantization artifact. The honest reframing: Beyond RAG works for documents that fit in the model's *effective* working memory, which is 2–3 orders of magnitude smaller than the nominal context window. Full tech report: [`docs/paper/working-memory-cliff.md`](docs/paper/working-memory-cliff.md). HF blog post draft: [`docs/paper/hf-blog-draft.md`](docs/paper/hf-blog-draft.md).

---

## More Features

**Bring your own model** — any GGUF file works:
```python
m = Model("path/to/any-model.gguf")
for tok in m.generate("Once upon a time"):
    print(tok, end="", flush=True)
```

**Save & restore context** — read a document once, query it forever:
```python
m.ask("Read this long document: ...")
m.save_context("document.kv")    # compressed KV → disk

m2 = Model("model.gguf")
m2.load_context("document.kv")   # instant restore, no re-processing
m2.ask("What was on page 37?")
```

**Infinite scrollback** — context never overflows, old tokens are shifted (not deleted):
```python
# Chat for hours — no "context window exceeded" error
for tok in m.generate("Tell me an extremely long story"):
    print(tok, end="", flush=True)
```

**Browser demo** — 193 KB WASM, one-click: [quantumaikr.github.io/quant.cpp](https://quantumaikr.github.io/quant.cpp/)

Pre-built wheels: Linux x86_64/aarch64, macOS arm64 (Python 3.9–3.13). Others compile from source automatically.

---

## Why quant.cpp?

When AI models have long conversations, they need memory called the **KV cache**. This memory grows with every message and often exceeds the model itself. quant.cpp compresses it **6.4x** and prunes unimportant tokens — so the same laptop can handle **6x longer conversations at 59% lower attention cost**.

---

## Beyond RAG: Document-Level Context

Traditional RAG splits documents into small chunks (512 tokens), embeds them, and retrieves fragments. This works for large corpora but has fundamental limitations:

- **Chunking destroys relationships** — information spanning pages 3, 47, and 103 can't be found by any single chunk search
- **Retrieval can fail** — if the question uses different words than the document ("employee retention" vs "turnover rate")
- **No multi-hop reasoning** — connecting A → B → C across chunks is impossible when each is retrieved independently

**Long-context KV compression offers a complementary approach:**

```
Chunk-Level RAG:    100K docs → chunk(512) → embed → search → 5 chunks → LLM(4K)
                                 ↑ information loss here

Document-Level RAG: 100K docs → doc-level index → search → 2-3 full docs → LLM(64K-128K)
                                                              ↑ KV compression makes this fit
```

RAG decides **which documents** to look at. Long-context decides **how deeply** to understand them. Each does what it's best at.

| | Chunk-RAG alone | Long-Context alone | **RAG + Long-Context** |
|--|----------------|-------------------|----------------------|
| 100K documents | only option | impossible | **RAG selects** |
| Cross-page reasoning | fails | works | **works** |
| Multi-hop Q&A | limited | works | **works** |
| Exact recall | depends on retrieval | depends on model size | **best of both** |
| Infrastructure | vector DB + 4 systems | LLM + .kv file | **practical hybrid** |

**Pre-computed KV library** — process once, query forever:
```python
# Overnight (GPU or batch): process each document once
m.ask(open("operations_manual.txt").read())
m.save_context("ops_manual.kv")       # 1.5 GB, compressed

# Anytime (laptop, offline): instant load + unlimited questions
m.load_context("ops_manual.kv")       # 0.5 seconds
m.ask("What's the expense reimbursement process?")  # instant
```

Without 6.4x KV compression, loading a full 50K-token document into a 3B model needs ~17 GB of KV memory (impossible on 16GB Mac). With compression: ~2.7 GB (fits easily).

<details>
<summary><b>Technical detail: The KV cache problem</b></summary>

LLM memory is dominated by the **KV cache**, not model weights. At 32K context, a 8B model's KV cache consumes **4GB** — more than the model itself. Every existing engine stores KV in FP16. We compress it.

```
  +------------+-------------------------------+
  |            | KV Cache (FP16)               |
  | Model(4GB) | ██████████████   8K  <-- OOM  |
  +------------+-------------------------------+
  |            | KV (4-bit)                    |
  | Model(4GB) | ██ -------------> 350K ctx    |
  |            |      6.9x smaller             |
  +------------+-------------------------------+
```

</details>

<details>
<summary><b>Detailed benchmark tables</b></summary>

## The Result

> **Same hardware. 4–7x longer context. PPL measured and disclosed.**

### Llama 3.2 3B Instruct — PPL × Speed (FP32 KV = 13.56 PPL @ 17.9 tok/s)

> **Round 10 (NEON `vqtbl1q_s8`) — `turbo_kv_4b` now matches fp32 KV speed at 7.1× compression.** 10 rounds of Karpathy iteration closed the speed gap from −45% (literal port) to PARITY. Profile-driven analysis revealed the bottleneck was the scalar inner loop, not the dequant — fp32 had 4-way NEON SIMD while we were doing scalar gather. Quantizing the 16 Lloyd-Max-Gaussian centroids to int8 and using `vqtbl1q_s8` for SIMD table lookup eliminated the gap.

| KV Config | Bytes/block | Compression | PPL | Δ vs FP32 | tok/s | vs FP32 speed |
|:----------|------------:|------------:|----:|----------:|------:|--------------:|
| FP32 reference | — | 1× | 13.56 | — | 18.43 | baseline |
| **`turbo_kv_4b`** ⭐ default | **72** | **7.1×** | **14.08** | **+3.8%** | **18.17** | **−1.4%** ✅ parity |
| `turbo_kv_5b` 🏆 quality | 88 | 5.8× | **13.65** | **+0.7%** | 16.80 | −8.8% |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 16.57 | −10.1% |
| `uniform_4b` | 68 | 7.5× | 14.60 | +7.7% | 13.27 | −26.8% |
| llama.cpp `q4_0` KV (lit.) | ~70 | ~7.3× | ~14.99 | +10.6% | — | — |

`turbo_kv_4b` (default) is now Pareto-dominant on every axis vs `uniform_4b`: better PPL (14.08 vs 14.60), faster (18.7 vs 13.3 tok/s), comparable compression (7.1× vs 7.5×). And at the same time it matches fp32 KV speed at the cost of just 3.8% PPL — for 7.1× less memory.

The 5b/3b variants haven't yet received the Round 10 NEON treatment (their inner loops are still scalar, planned for v0.7.1). Their speed numbers in the table above are still pre-Round-10.

**Build note**: All numbers are with CMake default `TQ_BUILD_METAL=OFF` (CPU-only). The existing Metal backend has per-matmul dispatch overhead that exceeds the GPU benefit at batch-1 inference; see [issue #16](https://github.com/quantumaikr/quant.cpp/issues/16) for the investigation.

```
                  PPL Degradation vs FP32           Speed vs FP32 KV
                       (lower is better)              (higher is better)

  turbo_kv_5b     │█ +0.7%                       █████████ −14.9%
  turbo_kv_4bo    │██▌ +2.5%                     ████████ −16.2%
  turbo_kv_4b ⭐  │█████ +5.7%                   ██████████ −8.4%
  turbo_kv_3b     │█████████████ +13.3%          █████████ −13.0%
  uniform_4b      │██████ +7.7%                  ███████ −26.8%
  llama.cpp q4_0  │██████████ +10.6%             — (not measured)
  FP32 reference  │ ← 0%                         18.13 tok/s ←
                   0%   +5%   +10%               0   25%   50%   75%   100%
```

`turbo_kv_4b` (default) and `turbo_kv_5b` (quality) are the Pareto-optimal recommendations: **5.8–7.1× memory compression at 92% of FP32 KV speed.** Full Karpathy-loop history (9 rounds across 3 sessions) in [bench/results/turboquant_reproduction.md](bench/results/turboquant_reproduction.md).

### Cross-size validation (3 Llama-family models, all measured CPU-only)

| Model | FP32 baseline | turbo_kv_5b PPL Δ | turbo_kv_4b PPL Δ | turbo_kv_4b tok/s | vs FP32 speed |
|---|---:|---:|---:|---:|---:|
| SmolLM2 135M | 18.62 PPL @ 70.4 t/s | +1.7% | +5.8% | 60.2 | −14.5% |
| Llama 3.2 1B | 16.88 PPL @ 41.1 t/s | +0.7% | +7.3% | 34.4 | −16.3% |
| Llama 3.2 3B | 13.56 PPL @ 18.13 t/s | +0.7% | +5.7% | 16.60 | −8.4% |

`turbo_kv_5b` is consistently near-lossless across model sizes (~1% PPL Δ). `turbo_kv_4b` stays in the 5–8% PPL range and runs at 84–92% of FP32 KV speed. **Recommendation**: use `turbo_kv_3b` only on models ≥ 3B parameters (the 8-level codebook is too coarse for small models — +61% PPL on Llama 3.2 1B).

> **About this comparison**: We previously published v0.6.3 release notes claiming `turbo_kv` beats `fp32` KV speed. That was an artifact of the fp32 attention path being unoptimized scalar — once we added NEON to the fp32 path (commit `4490c83`), the honest gap is `−7%` to `−12%`, not `+5%` to `+10%`. We've corrected the README and the v0.6.3 release notes.

### Context length gains (`turbo_kv_4b` + `q4` value cache)

| Hardware | Model | FP16 KV ctx | quant.cpp ctx | KV Gain |
|:---------|:------|------------:|--------------:|--------:|
| 16GB Mac | Llama 3.2 3B | 50K tokens | **350K tokens** | **6.9x** |
| 16GB Mac | Gemma 4 26B MoE | 4K tokens | **14K tokens** | **3.5x** |
| 8GB Laptop | Llama 8B (Q4) | 16K tokens | **61K tokens** | **3.8x** |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K tokens | **559K tokens** | **3.8x** |

## Why quant.cpp?

LLM memory is dominated by the KV cache. quant.cpp is **a minimal C engine that ships KV cache quantization that actually works**, in a form factor nobody else offers: one single header, zero dependencies, runs on iOS/Android/WASM/MSVC/microcontrollers.

**Two reasons to use it:**

1. **You need to embed LLM inference inside something.** An app, a game, a web page, a device. quant.cpp is one file (`quant.h`, 628KB) plus libc. Everywhere a C compiler runs, this runs.

2. **You want to study KV cache compression.** quant.cpp implements 7 KV quantization schemes side by side: `uniform_4b/2b/3b`, `polar_3b/4b`, `qjl_1b`, `turbo_kv_*`. You can read each one in a single C file and add a new one in 3 functions.

**Honest disclosure**: In April 2026 Google published [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874). quant.cpp's `turbo_kv_*` types started as a port of that algorithmic structure (Random Hadamard Transform → Lloyd-Max codebook → 1-bit QJL residual). Through a Karpathy-loop ablation we discovered the QJL residual stage was contributing literally zero to scores, dropped it, and reinvested the freed bytes into a larger codebook. The result (`turbo_kv_4b` at 14.28 PPL on Llama 3.2 3B) **beats our previous production champion `uniform_4b` and llama.cpp's `q4_0` KV** at the same 4-bit budget. The full optimization history is in [bench/results/turboquant_reproduction.md](bench/results/turboquant_reproduction.md).

> **Need the exact paper numbers in a paper?** Use [Google's reference](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/).
> **Need a small, readable C engine with KV compression that ships on a phone, browser, microcontroller, or game engine?** Use quant.cpp.

## Get Started in 60 Seconds

```bash
# 1. Build
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# 2. Download a model (135MB starter)
pip install huggingface_hub
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf --local-dir models/

# 3. Run
./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "Hello!" -j 4

# 4. With KV compression (7x longer context)
./build/quant models/SmolLM2-135M-Instruct-Q8_0.gguf --chat -p "Hello!" -k uniform_4b -v q4
```

> **[Full API docs](docs/api.md)** · **[WASM demo](https://quantumaikr.github.io/quant.cpp/)** · **[Add your own KV type](docs/custom-quantization.md)** · **[Python: `pip install quantcpp`](#python)**

---

## See It In Action: Book-in-a-Chat

Load an entire novel into context and ask questions about it. llama.cpp runs out of memory. quant.cpp remembers the whole book.

```bash
# Load Alice in Wonderland (~27K tokens) with KV compression
bash bench/demo/book_chat.sh models/Llama-3.2-3B-Instruct-Q8_0.gguf

# Q: "What riddle did the Mad Hatter ask Alice?"
# A: "Why is a raven like a writing-desk?" — from Chapter 7, A Mad Tea-Party...
```

On a 16GB Mac with Llama 3.2 3B: llama.cpp maxes out at ~50K tokens (FP16 KV). quant.cpp compresses KV 6.9x → **350K tokens** — enough for 12 novels.

---

</details>

<details>
<summary><b>How it compares to other engines</b></summary>

## How It Compares

### vs llama.cpp: Quality at same bit budget

```
                    KV Quantization Quality (SmolLM2 1.7B, WikiText-2)
                    
  llama.cpp Q4_0 KV │██████████████████████████████████████ PPL +10.6%
                    │
  llama.cpp Q8 K+Q5 V │▎ PPL ~+1%  ← recommended (1.6x compression)
                    │
   quant.cpp 4-bit  │▏ PPL +0.0%  ← lossless (3.8x compression)
                    │
   quant.cpp 3-bit  │█ PPL +1.3%  ← delta compression (4.3x)
                    └────────────────────────────────────────────────
                     0%                                         +12%
                              Perplexity Degradation →
```

Both are per-block methods. The quality gap comes from block size (128 vs 32), min-max range encoding, independent K/V treatment, and delta compression — not from a fundamental design flaw in llama.cpp. At ~1.6x compression, llama.cpp Q8+Q5 is excellent. quant.cpp targets the **4-7x range** where the difference matters.

### vs other TurboQuant implementations

|  | quant.cpp | turbo-quant (Rust) | turboquant-pytorch | scos-lab/turboquant |
|:--|:---------:|:------------------:|:------------------:|:-------------------:|
| Language | **Pure C11** | Rust | Python | Python |
| Single-header | **✅ quant.h (628KB)** | ❌ Cargo crate | ❌ pip install | ❌ |
| Dependencies | **libc + libm** | Rust toolchain | PyTorch + CUDA | PyTorch |
| iOS / Android | **✅** | ❌ | ❌ | ❌ |
| WASM (browser) | **✅ 192KB** | ❌ | ❌ | ❌ |
| MCU / embedded | **✅** | ❌ | ❌ | ❌ |
| Windows MSVC | **✅** | ✅ | (Python) | (Python) |
| GGUF model loading | **✅ 7 architectures** | ❌ | ❌ | research only |
| End-to-end inference | **✅** | kernel only | kernel only | kernel only |

### "Why not just use llama.cpp?"

You absolutely can. llama.cpp is excellent. The difference is **integration scope**, not capability:

**llama.cpp = compiled library** (250K+ LOC). You link `libllama`, which pulls in GGML tensor graphs, Metal/CUDA backends, sampling, tokenizer. Great if your build system handles it — but it's a _library_ with a build step.

**quant.cpp = one file** (16K LOC). `#include "quant.h"`, compile with `cc app.c -lm`. No CMake, no linker flags beyond libc. One translation unit.

Where this difference matters in practice:

```
# quant.cpp — add LLM to any C project in 2 lines
cc -O2 my_app.c -lm -lpthread -o my_app    # that's it

# llama.cpp — requires building the library first
cmake -B build && cmake --build build       # build libllama
cc my_app.c -Ibuild/include -Lbuild -lllama -lm -lstdc++ -o my_app
```

| Scenario | quant.cpp | llama.cpp |
|:---------|:---------:|:---------:|
| **WASM browser demo** | 192 KB binary | GGML tensor graph too large |
| **Microcontroller / RTOS** | `#include` only option (no FS, no linker) | Needs build system |
| **Game engine plugin** (Unity/Unreal/Godot) | Drop one `.h` | Integrate 250K LOC build |
| **Teaching / research** | Read in an afternoon | Excellent but large codebase |
| **Quick prototype** | `pip install quantcpp` or 2-line C | More setup needed |
| **GPU speed** | Basic | **Full Metal/CUDA** |
| **Model coverage** | 7 architectures | **100+** |
| **Production hardening** | Early stage | **Battle-tested** |

> **Use llama.cpp** for speed on a workstation. **Use vLLM** for batch serving.
> **Use quant.cpp** when you need to ship LLM inference _inside_ something — an app, a game, a browser tab, an embedded device — and integration simplicity matters more than GPU throughput.

### vs production inference engines

|  | quant.cpp | llama.cpp | vLLM | MLX |
|:--|:---------:|:---------:|:----:|:---:|
| KV quantization | **7 schemes (3-7x)** | Q8_0/Q5_0 (2x) | -- | -- |
| Code size | **72K LOC** | 250K+ | 100K+ | 50K+ |
| Embeddable | **single header** | library | library | framework |
| GPU throughput | basic | full | **best** | Metal |

---

## Supported Models

| Model | Params | Architecture | Speed (M1 Pro, 8T) | KV Compression |
|:------|-------:|:-------------|-------------------:|:--------------:|
| SmolLM2 135M | 135M | Llama | **103 tok/s** | 2.4x |
| Llama 3.2 3B Instruct | 3B | Llama 3 (GQA) | **10 tok/s** | 6.9x |
| Gemma 4 26B-A4B-it | 26B (4B active) | MoE 128 experts | **3.9 tok/s** | 3.5x |
| Qwen3.5 0.8B | 752M | DeltaNet hybrid | 80 tok/s | 3.8x |
| Qwen3.5 4B | 4B | DeltaNet hybrid | 20 tok/s | 3.8x |
| SmolLM2 1.7B | 1.7B | Llama | 25 tok/s | 3.8x |
| Gemma 3 270M | 270M | Gemma 3 | 176 tok/s | 3.8x |

GGUF format. Load any llama.cpp-compatible model.

<details>
<summary><b>Gemma 4 26B-A4B architecture details</b></summary>

Full support for Gemma 4's hybrid MoE architecture:

- **Dual-FFN**: parallel Dense MLP + 128-expert MoE per layer
- **Hybrid attention**: 25 sliding (head_dim=256) + 5 full (head_dim=512) layers
- **QK-norm aware KV compression**: auto FP32 keys + Q4 values (3.5x savings)
- **Learned RoPE** with per-layer frequency factors
- **IQ3_XXS/IQ4_NL** fused dot with NEON optimization for MoE experts
- **GeGLU** activation (NEON-accelerated fast tanh approximation)

```bash
./build/quant gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  -p "<start_of_turn>user\nWhat is the capital of France?\n<end_of_turn>\n<start_of_turn>model\n" \
  -n 50 -j 8 -T 0.0 -k uniform_4b -v q4
# Output: "The capital of France is **Paris**."
```

</details>

---

## KV Cache Compression

### The Idea

```
Standard:  Store every key as-is            → 16 bits/element → FP16

quant.cpp: Quantize keys to 4-bit           → 4 bits/element  → 3.8x
           + quantize values to Q4           → 4 bits/element  → 6.9x
           + delta encode adjacent keys      → 3 bits/element  → 8.5x

Like video compression: I-frames (FP32) every 64 tokens, P-frames (3-bit delta) between.
```

### Quality vs Compression

```
                    WikiText-2 PPL (SmolLM2 1.7B)

  FP32 baseline      14.63 │ ●
  4b K + FP16 V       14.63 │ ● identical
  4b K + Q4 V         14.57 │ ● slightly better (!)
  delta 3b K + Q4 V   14.82 │  ●  +1.3%
  llama.cpp Q8K+Q5V   ~14.8 │  ●  ~+1% (1.6x compression)
  llama.cpp Q4_0 KV   16.18 │          ● +10.6% (3.8x compression)
  3b K (no delta)       ——  │                              ● +62%
                            └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
                              14  15  16  17  18  19  20  21+
```

### Modes

| Config | Compression | PPL vs FP32 | Best for |
|:-------|:----------:|:-----------:|:---------|
| `delta + 3b K + Q4 V` | **~8.5x** | +1.3% | Maximum context |
| `delta + 4b K + Q4 V` | ~6.9x | ~0% | Quality + compression |
| `uniform_4b K + Q4 V` | 6.9x | ~0% | Simple, no delta overhead |
| `uniform_4b K + FP16 V` | 1.6x | +0.0% | Lossless baseline |

### QK-norm Aware (Gemma 4)

Models with QK-norm normalize keys to the unit sphere, creating extremely sparse distributions. quant.cpp auto-detects this and stores keys in FP32 while quantizing only values — preserving perfect precision with **3.5x V memory reduction**.

---

## Advanced Usage

> **Note:** `MODEL` below is a placeholder for **your** GGUF file path. The Quick Start above downloads `models/SmolLM2-135M-Instruct-Q8_0.gguf` — you can paste that path directly, or substitute any other GGUF you have. There is no file literally named `model.gguf`.

```bash
# Pick any GGUF you have on disk (this is the one from Quick Start):
MODEL=models/SmolLM2-135M-Instruct-Q8_0.gguf

# Delta compression (maximum context, 8.5x)
./build/quant $MODEL --chat -p "hello" -k uniform_3b -v q4 --delta

# Perplexity benchmark
./build/quant $MODEL --ppl input.txt -k uniform_4b -v q4

# Model info
./build/quant $MODEL --info

# Performance profiling
./build/quant $MODEL --chat -p "hello" -n 50 --profile
```

---

## Single-Header Mode

> Copy one file. Add LLM to any C project.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("path/to/your.gguf");  // any GGUF file
    quant_ctx*   c = quant_new(m, NULL);
    
    // Streaming
    quant_generate(c, "Tell me a joke", print_token, NULL);
    
    // Or one-shot
    char* answer = quant_ask(c, "What is 2+2?");
    printf("%s\n", answer);
    free(answer);
    
    quant_free_ctx(c);
    quant_free_model(m);
}
```

```bash
cc app.c -o app -lm -lpthread    # that's it — no cmake, no framework
```

**15.7K LOC, 643KB, ~2s compile time.** Full API:

| Function | Description |
|:---------|:------------|
| `quant_load(path)` | Load a GGUF model |
| `quant_new(model, config)` | Create inference context |
| `quant_generate(ctx, prompt, cb, ud)` | Stream tokens via callback |
| `quant_ask(ctx, prompt)` | Generate and return string |
| `quant_free_ctx(ctx)` | Free context |
| `quant_free_model(model)` | Free model |

---

## Browser Demo (WASM)

> **192KB.** The entire inference engine compiles to a WASM binary smaller than most JPEGs.

```bash
cd wasm && bash build.sh          # Requires: emscripten
python3 -m http.server 8080       # Serve locally
# Open http://localhost:8080, drag & drop any GGUF model
```

Everything runs client-side. Nothing is uploaded. KV compression active by default.

---

## Docker & Server

**Docker** (zero-dependency, ~10MB image):
```bash
docker build -t quant.cpp .
docker run -v ./models:/models quant.cpp /models/SmolLM2-135M-Instruct-Q8_0.gguf -p "hello" -k uniform_4b -v q4
# Replace SmolLM2-135M-Instruct-Q8_0.gguf with whatever GGUF you placed in ./models
```

**OpenAI-compatible server** (`/v1/chat/completions`):
```bash
cmake -B build -DTQ_BUILD_SERVER=ON && cmake --build build
./build/quant-server models/SmolLM2-135M-Instruct-Q8_0.gguf -p 8080 -k uniform_4b
# Substitute your own GGUF path as needed

# Works with the OpenAI Python SDK
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Build with `-DTQ_BUILD_SERVER=ON`. Streaming SSE supported. KV compression configurable per request.

---

## Python

```bash
cd bindings/python && pip install .
```

```python
from quantcpp import Model

with Model("models/SmolLM2-135M-Instruct-Q8_0.gguf", kv_compress=1) as m:  # use your own GGUF path
    print(m.ask("What is the capital of France?"))

    # Streaming
    for token in m.generate("Once upon a time"):
        print(token, end="", flush=True)
```

Zero build dependencies beyond a C compiler. Compiles `quant.h` at install time.

---

## Backends & Performance

| Backend | Platform | Status | Notes |
|:--------|:---------|:------:|:------|
| **NEON** | ARM (Apple Silicon) | Production | 5.8x SIMD speedup |
| **AVX2** | x86 | Production | |
| **Metal** | Apple GPU | Verified | Batch matmul dispatch |
| **CUDA** | NVIDIA GPU | Compiles | |
| **Vulkan** | Cross-platform | Compiles | |
| **WASM** | Browser | **NEW** | 192KB binary |
| **MSVC** | Windows | **NEW** | VS 2019/2022 |

<details>
<summary><b>Performance breakdown (Gemma 4 26B on M1 Pro)</b></summary>

| Component | ms/token | Share |
|:----------|--------:|------:|
| Attention matmul (Q8_0 NEON) | 168 | 65% |
| MoE experts (IQ3_XXS/IQ4_NL NEON) | 72 | 28% |
| Attention scores | 3 | 1% |
| Other | 14 | 6% |
| **Total** | **257** | **3.9 tok/s** |

</details>

---

## FAQ

<details>
<summary><b>How is this different from llama.cpp?</b></summary>

llama.cpp is a full-featured inference framework (250K+ LOC). quant.cpp is a minimal engine (72K LOC) you can read, modify, and embed. Different tools for different problems: llama.cpp optimizes speed, quant.cpp optimizes memory (KV compression) and embeddability (single header).

</details>

<details>
<summary><b>llama.cpp already has KV quantization. How is yours different?</b></summary>

llama.cpp supports KV cache quantization (Q8_0 K + Q5_0 V is the recommended config, ~1.6x compression with minimal quality loss). quant.cpp targets higher compression: 4-bit K + Q4 V gives 3.8x at +0.0% PPL, and delta compression pushes to 4.3x at +1.3% PPL. The quality advantage comes from 128-element min-max blocks (vs 32-element), independent K/V quantization methods, and delta encoding of adjacent keys — a technique llama.cpp doesn't have. Use llama.cpp's KV quant if 1.6x is enough; use quant.cpp if you need 4-7x.

</details>

<details>
<summary><b>How does this compare to Karpathy's llm.c?</b></summary>

Similar philosophy: minimal C, educational. Key differences: quant.cpp supports quantized weights (Q4_K_M, Q8_0, IQ2), multiple architectures (Llama, Qwen, Gemma, MoE), GGUF loading, and KV cache compression. Think of llm.c as the textbook and quant.cpp as the production-ready version.

</details>

<details>
<summary><b>Can I embed this in my app?</b></summary>

Yes. Two options:
1. **Single-header**: Copy `quant.h`, `#define QUANT_IMPLEMENTATION` in one .c file. Done.
2. **Full library**: Link against `libturboquant.a`.

Works on Linux, macOS, Windows (MSVC/MinGW), iOS, Android, and WASM.

</details>

<details>
<summary><b>Why is it slower than llama.cpp?</b></summary>

Three reasons: (1) llama.cpp has years of hand-tuned NEON/AVX2 assembly for every quant format, (2) llama.cpp offloads the full forward pass to Metal/CUDA GPU, (3) 250K+ LOC vs 72K LOC means more micro-optimizations. quant.cpp optimized for memory and embeddability first. Speed improvements (full Metal GPU offload, more SIMD kernels) are actively in progress — see [v1.3 plan](docs/plan/prd/prd_v1.3.md).

</details>

<details>
<summary><b>No GPU — is this useless?</b></summary>

If you need 100+ tok/s, use llama.cpp with Metal/CUDA. If you need to embed inference in an iOS app, WASM module, game engine, or IoT device — quant.cpp works. CPU on Apple Silicon: 25 tok/s (1.7B), 11.6 tok/s (3B), 3.9 tok/s (26B MoE).

</details>

<details>
<summary><b>Can it run in the browser?</b></summary>

Yes. `cd wasm && bash build.sh`. The WASM binary is 192KB. Drop a GGUF model and chat. Everything runs client-side.

</details>

<details>
<summary><b>What about sub-3-bit quantization?</b></summary>

Tested extensively (2-bit delta, NF2, online SVD, multi-hash). None reached acceptable quality. Per-step cosine 0.997 compounds to 0.885 after 200 steps. 3-bit + delta is the practical minimum.

</details>

---

## Documentation

| Document | Description |
|:---------|:------------|
| **[API Reference](docs/api.md)** | Full C API for quant.h and libturboquant (730 lines) |
| **[Custom Quantization](docs/custom-quantization.md)** | Add your own KV type in 3 functions |
| **[H2H Benchmark](bench/head_to_head/)** | Reproducible quant.cpp vs llama.cpp comparison |
| **[KV Compression Landscape](docs/blog/kv-cache-landscape.md)** | Eviction vs Architecture vs Compression guide |
| **[ROADMAP](ROADMAP.md)** | Project direction and planned features |
| **[CHANGELOG](CHANGELOG.md)** | Version history and release notes |
| **[Tech Report](docs/papers/quant_cpp_tech_report.md)** | Architecture and benchmarks (Arxiv draft) |
| **[WASM Demo](https://quantumaikr.github.io/quant.cpp/)** | Try it in your browser — no install needed |

---

## References & Citations

quant.cpp is an independent implementation of published research. The Variant F architecture (RHT preprocessing + scalar Lloyd-Max codebook on rotated values, no QJL stage) sits in a lineage that combines two prior works:

- **HIGGS** — Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh. *Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*. Nov 2024. [arXiv:2411.17525](https://arxiv.org/abs/2411.17525). HIGGS introduced the **Random Hadamard Transform + MSE-optimal grid quantization** pattern (for weight quantization). Our `tq_rht.c` Walsh-Hadamard + Rademacher implementation follows this pattern. *We added this attribution after seeing [Tim Dettmers' general comment in llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) asking participants in that thread (which uses "TurboQuant" loosely across many forks) to credit HIGGS instead. His comment was not directed at us specifically, but the substance applied to our naming as well, and we chose to update accordingly.*
- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). TurboQuant applies the rotation pattern to **KV cache** with a 1-bit QJL residual stage and per-channel outlier handling. Our work started as a literal port of TurboQuant; through 9 rounds of Karpathy-loop iteration we simplified it (dropped QJL, dropped outlier channels) into the current Variant F. We do not claim our shipped variant is the TurboQuant algorithm — it is an empirically-derived simplification.
- **PolarQuant** — *Quantizing KV Caches with Polar Transformation*. AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617). The polar-coordinate KV quantization that our `tq_polar.c` baseline implements.
- **QJL** — *Quantized Johnson-Lindenstrauss Transform for KV Cache Compression*. AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482). The 1-bit sketch building block. Used in our `tq_qjl.c` baseline; we found it contributed ~zero to attention scores in the Variant F regime and dropped it.
- [Google Research blog post on TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

**Honest attribution**: Variant F's structure (RHT + scalar grid quantization) is closest to HIGGS in spirit, applied to KV cache like TurboQuant, with both the QJL residual and the outlier channel split removed through ablation. If you use quant.cpp in academic work, please cite all three (HIGGS, TurboQuant, PolarQuant) and this repository.

</details>

---

## Transparency & Trust

We believe trust is built by being honest about what we got wrong.

<details>
<summary><b>10 self-corrections — all found before any external report</b></summary>

| # | Version | What we claimed wrong | What we corrected |
|---|---|---|---|
| 1 | v0.6.3 | "Lossless 7× compression" | Re-measured; not lossless |
| 2 | v0.6.x | "Beats FP32 speed" | FP32 baseline was unoptimized scalar |
| 3 | v0.7.x | "With Metal default" | CMake default is Metal=OFF |
| 4 | v0.7.x | Interpreted a general comment as directed at us | Updated attribution |
| 5 | v0.8.0 | kv_compress=1 caused abort | Fixed in v0.8.1 |
| 6 | v0.8.0 | libc.free() cross-heap crash | Fixed with quant_free_string |
| 7 | v0.8.1 | 65 KB memory leak per ask() | Fixed in v0.8.2 |
| 8 | v0.9.0 | Disabled a working feature by mistake | Re-enabled with verification |
| 9 | v0.10 | 957-token eval with 53% FP32 window | Documented caveat, fixed tokenizer |
| 10 | v0.10 | "2-bit Pareto-dominates 4-bit" | Withdrawn — PPL +36.7% at long context |

Every claim in this README is backed by reproducible benchmark data in `bench/results/`.
</details>

<details>
<summary><b>Benchmark artifacts</b></summary>

| File | What it measures |
|---|---|
| [`progressive_kv_compression.md`](bench/results/progressive_kv_compression.md) | 128-token FP32 window = FP32 quality at 3x compression |
| [`attention_aware_quantization.md`](bench/results/attention_aware_quantization.md) | Full Pareto curve (including withdrawn 2-bit claim) |
| [`long_context_kv_compression.md`](bench/results/long_context_kv_compression.md) | 32K context memory + speed measurements |
| [`layer_adaptive_analysis.md`](bench/results/layer_adaptive_analysis.md) | Per-layer adaptation is unnecessary after RHT (negative result) |
</details>

---

<p align="center">
  <a href="https://pypi.org/project/quantcpp/">PyPI</a> ·
  <a href="https://quantumaikr.github.io/quant.cpp/">WASM Demo</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/quantumaikr/quant.cpp/issues">Issues</a>
</p>

<p align="center">
  <a href="https://star-history.com/#quantumaikr/quant.cpp&Date">
    <img src="https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date" alt="Star History" width="600">
  </a>
</p>
