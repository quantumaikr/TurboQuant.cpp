# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**LLM inference engine with extreme KV cache compression. Zero dependencies. Pure C.**

Load a model, generate text, compress KV cache — all in one binary, no Python needed.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-70%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Qwen3.5](https://img.shields.io/badge/Qwen3.5--0.8B-14%20tok%2Fs-blue)]()

---

## At a Glance

| | PyTorch | TurboQuant.cpp |
|---|---|---|
| **CPU Speed** | 0.8 tok/s | **14 tok/s** (17x) |
| **GPU Speed** | 10 tok/s (MPS) | **14 tok/s (CPU only!)** |
| **Weight Memory** | 1.7 GB (BF16) | **533 MB** (Q8, `-q` flag) |
| **KV Cache** | FP16 (full size) | **7.5x compressed** (4-bit) |
| **Dependencies** | PyTorch + transformers | **0** (pure C) |

> Qwen3.5-0.8B on Apple Silicon. Our CPU engine beats PyTorch on GPU.

---

## Run It

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Generate text (14 tok/s, Q8 weights, 4 threads)
./build/tq_run model.safetensors -t tokenizer.json -p "What is AI?" -j 4 -q
```

```
Prompt: What is AI?
---
Artificial intelligence (AI) is a field of computer science that focuses
on creating systems capable of performing tasks that typically require
human intelligence...
---
100 tokens in 7.2s (13.9 tok/s, 4 threads, kv=uniform_4b)
```

### Python

```python
from turboquant import TurboQuant
tq = TurboQuant("cpu")
compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)  # 7.5x smaller
scores = tq.attention(query, compressed, seq_len, dim, TurboQuant.UNIFORM_4B)
```

---

## What Makes It Fast

### 1. Self-Contained Engine

Not a wrapper — a full inference engine in pure C:

```
Model Loading    safetensors (mmap, BF16→FP32 streaming)
Tokenizer        HuggingFace BPE (248K vocab)
Forward Pass     DeltaNet + Self-Attention (Qwen3.5 hybrid)
KV Cache         TurboQuant quantized (4-bit, auto-compressed)
Attention        Integer Q4×Q8 (2.9x faster than FP32)
Weights          Q8 quantization (-q flag, 4x memory savings)
Generation       Top-p sampling, streaming output
```

### 2. Integer-Domain Attention

Attention scores computed directly on quantized data — no dequantization:

```
FP32 attention:  22.8 μs (baseline)
Q4×Q8 integer:    7.8 μs (2.9x faster, ARM vdotq_s32)
```

### 3. Q8 Weight Quantization

Weights compressed 4x with negligible quality loss:

```
./build/tq_run model.safetensors -p "1+1=" -q
→ "2" (correct, 533 MB instead of 2.1 GB)
```

---

## Real Model Validation

Tested on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — real inference, not synthetic:

| Test | Result |
|------|--------|
| "1+1=" | **2** ✓ |
| "The capital of France is" | **Paris** ✓ |
| "The capital of Japan is" | **Tokyo** ✓ |
| "What is deep learning?" | Correct paragraph ✓ |
| Logits cosine vs PyTorch | **0.999** |

### KV Cache Quality

| Type | Compression | Quality (Cosine) | Grade |
|------|-------------|-----------------|-------|
| **uniform_4b** | 7.5x | 0.994 | **A+** |
| **mixed_4b8** | 6.4x | 0.994 | **A+** |
| uniform_2b | 14.2x | 0.953 | A |

---

## CLI Reference

```bash
# Basic inference
./build/tq_run MODEL -t TOKENIZER -p "prompt" -n 100

# Options
-j 4          # threads (default: 4)
-q            # Q8 weight quantization (4x memory savings)
-k uniform_4b # KV cache type
-T 0.7        # temperature
-P 0.9        # top-p
--info         # show model info and exit
```

### Python CLI

```bash
python3 tools/tq info                          # quantization types
python3 tools/tq bench                         # performance benchmark
python3 tools/tq +memory llama-3.2-3b 65536    # memory calculator
python3 tools/tq +memory qwen3.5-0.8b 131072 --json  # JSON output
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started.md)** | Build, run, integrate |
| [Architecture](docs/architecture.md) | Engine design, type system |
| [Qwen3.5 Validation](docs/qwen35_validation_results.md) | Real model A/B results |
| [Integration Guide](docs/integration_guide.md) | llama.cpp, vLLM, Python |
| [Changelog](CHANGELOG.md) | Release notes |

---

## Technical Summary

- **Self-contained inference** — model load, tokenize, forward, generate in pure C
- **8 quantization types** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **Q8 weights** — 4x memory reduction, NEON-optimized matmul
- **Integer attention** — Q4×Q8 via ARM `vdotq_s32`
- **Multi-threaded** — pthread matmul, configurable threads
- **Hybrid model** — DeltaNet (recurrent) + Self-Attention (Qwen3.5)
- **RHT** — Random Hadamard Transform for 3.9x MSE reduction
- **K/V asymmetric** — independent key/value bit allocation
- **Zero dependencies** — pure C11, libc/libm only
- **70+ tests** — 19 C++ suites + 22 Python, ASan/UBSan/TSan clean

---

## References

- **TurboQuant** — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **QJL** — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- **PolarQuant** — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

---

**Developed by [QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
