# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**Multi-architecture LLM inference engine in pure C. Zero dependencies.**

Qwen3.5 + Gemma 3 supported. Gemma 4 ready.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-70%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Qwen3.5](https://img.shields.io/badge/Qwen3.5--0.8B-82%20tok%2Fs-blue)]()
[![Gemma3](https://img.shields.io/badge/Gemma3--270M-176%20tok%2Fs-blue)]()

### Supported Models

| Model | Params | Speed (Q4, 6T) | Verified |
|-------|--------|----------------|----------|
| **Qwen3.5-0.8B** | 752M | 82 tok/s | logits 0.999 cosine vs PyTorch |
| **Gemma 3 270M** | 270M | 176 tok/s | per-layer exact match vs PyTorch |

### llama.cpp vs TurboQuant — Fair Q4 Benchmark

```
Qwen3.5-0.8B, Q4_0, CPU-only, Apple Silicon M-series
─────────────────────────────────────────────────────
Threads │ llama.cpp  │ TurboQuant │
────────┼────────────┼────────────┤
   1    │  50.7 t/s  │  51.1 t/s  │ ← matched
   2    │  80.6 t/s  │  75.4 t/s  │
   4    │  90.0 t/s  │  71.6 t/s  │
   6    │     —      │  81.8 t/s  │ ← peak
```

Same model, same quantization, same hardware. Apples-to-apples.

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

That's it. The script builds the engine, downloads [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (~1.5 GB), converts it to TQM, and runs inference.

<details>
<summary>Manual setup (if you prefer step by step)</summary>

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
pip3 install huggingface_hub && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B')"
./build/tq_convert -o model.tqm
./build/tq_run model.tqm -p "What is deep learning?" -j 4
```
</details>

```
Prompt: What is deep learning?
---
Deep learning is a field of artificial intelligence and machine learning
that uses artificial neural networks to learn complex patterns...
---
100 tokens in 1.2s (81.8 tok/s, 6 threads, weights=Q4, kv=uniform_4b)
```

---

## Why TurboQuant?

|  | llama.cpp (Q4) | TurboQuant.cpp (Q4) |
|---|---|---|
| **Speed (1T)** | 50.7 tok/s | **51.1 tok/s** |
| **Loading** | ~1 sec | **0.3 sec** (mmap) |
| **KV Cache** | Full size | **7.5x compressed** |
| **Dependencies** | cmake, ggml | **None** (libc only) |
| **Quality** | Baseline | **0.999 cosine** (vs PyTorch F32) |
| **Unique** | Broad model support | **KV cache compression** |

---

## What's Inside

```
┌─────────────────────────────────────────────────────┐
│  tq_convert                                          │
│    safetensors → TQM (pre-quantized, mmap-ready)    │
├─────────────────────────────────────────────────────┤
│  tq_run                                              │
│    TQM → mmap load → forward → stream tokens        │
│                                                      │
│    ┌─── Architecture Dispatch ─────────────────┐   │
│    │  Qwen3.5: DeltaNet + Self-Attention + SwiGLU│  │
│    │  Gemma 3: Sliding Window + GQA + GeGLU      │  │
│    │  KV Cache: TurboQuant Q4 quantized          │  │
│    │  Attention: Integer Q4×Q8 (2.9x vs FP32)   │  │
│    └─────────────────────────────────────────────┘  │
│                                                      │
│    Q4 Weights ─── NEON matmul ─── Thread pool       │
└─────────────────────────────────────────────────────┘
```

### The Five Optimizations

| # | Technique | Impact |
|---|-----------|--------|
| 1 | **Q4 weights** — 4-bit quantized, 8x smaller | 2x faster (less data to read) |
| 2 | **TQM format** — pre-quantized mmap | 10x faster loading |
| 3 | **Integer attention** — Q4×Q8 via ARM vdotq_s32 | 2.9x faster attention |
| 4 | **Thread pool** — zero-overhead dispatch, NEON 2-row batch | 1.6x faster |
| 5 | **lm_head Q4** — output projection quantized at load time | 2x faster logits |

### Real Model Validated

Both architectures verified against PyTorch — actual inference, not synthetic:

```
Qwen3.5-0.8B:
  "1+1="                    → "2"                    ✓
  "What is deep learning?"  → correct paragraph      ✓
  Logits cosine vs PyTorch  → 0.999                  ✓

Gemma 3 270M:
  "1+1="                    → "2"                    ✓
  Forward pass              → per-layer exact match   ✓
  176 tok/s (Q4, 6 threads)                           ✓
```

---

## Speed Across Thread Counts

```
Qwen3.5-0.8B Q4, 100 tokens, CPU-only
──────    ──────────   ──────────────
Threads   Speed        vs llama.cpp
──────    ──────────   ──────────────
1         51.1 tok/s   1.01x ✓
2         75.4 tok/s   0.94x
4         71.6 tok/s   0.80x
6         81.8 tok/s   peak
8         77.5 tok/s
```

---

## CLI

```bash
# Convert (one-time)
./build/tq_convert                     # auto-detect model
./build/tq_convert model.safetensors tokenizer.json -o model.tqm

# Inference
./build/tq_run model.tqm -p "Hello"    # tokenizer embedded
./build/tq_run model.tqm -p "Hello" -j 4 -n 200 -T 0.7

# Python CLI
python3 tools/tq info                  # quantization types
python3 tools/tq +memory llama-3.2-3b 65536
python3 tools/tq_chat.py "What is AI?" # native engine + KV analysis
```

### Python API

```python
from turboquant import TurboQuant
tq = TurboQuant("cpu")
compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)  # 7.5x
scores = tq.attention(query, compressed, seq_len, dim, TurboQuant.UNIFORM_4B)
```

---

## Documentation

| Doc | What's in it |
|-----|-------------|
| **[Getting Started](docs/getting-started.md)** | Build, convert, run, integrate |
| [Architecture](docs/architecture.md) | Engine design, 4-layer stack |
| [Qwen3.5 Results](docs/qwen35_validation_results.md) | Real model A/B tests |
| [Changelog](CHANGELOG.md) | Full version history |
| [Integration](docs/integration_guide.md) | llama.cpp, vLLM, Python |

---

## Under the Hood

- **Multi-architecture** — Qwen3.5 (DeltaNet hybrid) + Gemma 3 (sliding window), Gemma 4 ready
- **9,000+ lines of C** — complete inference engine, no wrappers
- **8 quantization types** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **TQM format** — pre-quantized binary model, mmap instant load
- **Dual tokenizer** — GPT2 byte-level BPE + SentencePiece auto-detect
- **Q4×Q8 integer attention** — ARM vdotq_s32, no float dequantization
- **Thread pool** — zero-overhead dispatch with NEON 2-row batching
- **20 test suites, 70+ tests** — ASan + UBSan + TSan clean

---

## The Journey

```
Day 1 morning:   Empty directory
Day 1 noon:      KV cache compression library (8 types, A/B tested)
Day 1 evening:   Full inference engine (model load → generate)
Day 1 night:     82 tok/s, matching llama.cpp on single-thread
Day 2:           Gemma 3 support, multi-architecture engine

Lines of C:      9,000+
Test suites:     20 (70+ tests)
Architectures:   Qwen3.5 + Gemma 3 (Gemma 4 ready)
Speed:           82 tok/s (Qwen3.5), 176 tok/s (Gemma3)
```

---

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization

Architecture inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), and [ONNX](https://github.com/onnx/onnx).

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
