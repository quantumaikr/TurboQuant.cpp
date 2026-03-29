# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**LLM inference engine in pure C. 47 tok/s. Zero dependencies.**

Load → Generate → Done. No Python. No GPU. Just one binary.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-70%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Speed](https://img.shields.io/badge/47%20tok%2Fs-Qwen3.5--0.8B-blue)]()

```
PyTorch CPU:     0.8 tok/s
PyTorch GPU:      10 tok/s
TurboQuant CPU:   47 tok/s  ← 59x faster, no GPU needed
```

---

## 30-Second Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Convert model (one-time, auto-detects from HuggingFace cache)
./build/tq_convert

# Run
./build/tq_run model.tqm -p "What is deep learning?" -j 4
```

```
Prompt: What is deep learning?
---
Deep learning is a field of artificial intelligence and machine learning
that uses artificial neural networks to learn complex patterns...
---
100 tokens in 2.1s (46.9 tok/s, 4 threads, weights=Q4, kv=uniform_4b)
```

---

## Why TurboQuant?

|  | PyTorch | TurboQuant.cpp |
|---|---|---|
| **Speed** | 0.8 tok/s | **47 tok/s** (59x) |
| **Loading** | 3 sec | **0.3 sec** (mmap) |
| **Weight Memory** | 1.7 GB | **270 MB** (Q4) |
| **KV Cache** | Full size | **7.5x compressed** |
| **Dependencies** | PyTorch, transformers, torch | **None** |
| **Binary Size** | ~2 GB installed | **~1 MB** |
| **Quality** | Baseline | **0.999 cosine** (vs PyTorch) |

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
│    ┌─── Forward Pass ────────────────────────────┐  │
│    │  DeltaNet (18 layers, recurrent)            │  │
│    │  Self-Attention (6 layers, GQA + RoPE)      │  │
│    │  SwiGLU FFN (all 24 layers)                 │  │
│    │  KV Cache: TurboQuant Q4 quantized          │  │
│    │  Attention: Integer Q4×Q8 (2.9x vs FP32)   │  │
│    └─────────────────────────────────────────────┘  │
│                                                      │
│    Q4 Weights ─── NEON matmul ─── Multi-threaded    │
└─────────────────────────────────────────────────────┘
```

### The Five Optimizations

| # | Technique | Impact |
|---|-----------|--------|
| 1 | **Q4 weights** — 4-bit quantized, 8x smaller | 2x faster (less data to read) |
| 2 | **TQM format** — pre-quantized mmap | 10x faster loading |
| 3 | **Integer attention** — Q4×Q8 via ARM vdotq_s32 | 2.9x faster attention |
| 4 | **Multi-threaded matmul** — pthread, NEON | 1.6x faster |
| 5 | **Streaming BF16** — embed on-demand, no bulk convert | 6x less memory |

### Real Model Validated

Tested on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — actual inference, not synthetic:

```
"1+1="                      → "2"                    ✓
"The capital of France is"  → "Paris"                ✓
"What is deep learning?"    → correct paragraph      ✓
Logits cosine vs PyTorch    → 0.999                  ✓
```

---

## Speed Across Sequence Lengths

```
Tokens    Speed       Note
──────    ─────────   ──────────────────
10        12 tok/s    first-token latency included
30        41 tok/s    ← 40 tok/s crossed
50        44 tok/s
100       47 tok/s    ← steady state
200       48 tok/s    ← peak
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

- **8,500+ lines of C** — complete inference engine, no wrappers
- **8 quantization types** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **TQM format** — pre-quantized binary model, mmap instant load
- **DeltaNet + Self-Attention** — Qwen3.5 hybrid architecture in pure C
- **BPE tokenizer** — HuggingFace compatible (248K vocab, embedded in TQM)
- **Q4×Q8 integer attention** — ARM vdotq_s32, no float dequantization
- **Multi-threaded** — pthread matmul with NEON, configurable threads
- **Repetition penalty** — prevents degenerate output loops
- **20 test suites, 70+ tests** — ASan + UBSan + TSan clean

---

## The Journey

```
Day 1 morning:   Empty directory
Day 1 noon:      KV cache compression library (8 types, A/B tested)
Day 1 evening:   Full inference engine (model load → generate)
Day 1 night:     47 tok/s, Q4 weights, TQM instant loading

Lines of C:      8,500+
Test suites:     20 (70+ tests)
Commits:         52
Speed:           0.8 → 47 tok/s (59x improvement)
```

---

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization

Architecture inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), and [ONNX](https://github.com/onnx/onnx).

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
