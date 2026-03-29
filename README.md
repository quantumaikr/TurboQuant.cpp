# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**LLM inference engine with extreme KV cache compression. Zero dependencies. Pure C.**

**14 tok/s on CPU** — 17x faster than PyTorch, 3x faster than PyTorch+GPU.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-41%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Qwen3.5-0.8B](https://img.shields.io/badge/Qwen3.5--0.8B-14%20tok%2Fs-blue)]()

---

## Results at a Glance

| | PyTorch | TurboQuant.cpp |
|---|---|---|
| **Inference Speed (CPU)** | 0.8 tok/s | **14 tok/s** (17x faster) |
| **Inference Speed (GPU)** | 10 tok/s (MPS) | **14 tok/s (CPU only!)** |
| **KV Cache Size** | 7.00 GB | **0.93 GB** (87% saved) |
| **Dependencies** | PyTorch + transformers | **0** (pure C) |
| **Quality** | 1.000 | **0.994** (A+) |

> Measured on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). Our CPU-only engine is faster than PyTorch on Apple GPU.

---

## Try It Now (30 seconds)

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp
cd TurboQuant.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# See the A/B comparison yourself
./build/ab_test

# Memory savings for real LLM models
./build/demo_real_model

# Speed: Integer attention vs FP32
./build/speed_int_vs_float
```

### Python

```bash
pip install -e bindings/python

python3 examples/python_quickstart.py
```

```python
from turboquant import TurboQuant
import numpy as np

tq = TurboQuant("cpu")
keys = np.random.randn(512, 128).astype(np.float32) * 0.15

compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)
print(f"Compressed: {keys.nbytes:,} → {len(compressed):,} bytes ({keys.nbytes/len(compressed):.1f}x)")
```

### C

```c
#include "turboquant/turboquant.h"

tq_context_t* ctx;
tq_init(&ctx, TQ_BACKEND_CPU);

// 7.5x compression, one line
tq_quantize_keys(ctx, keys, n, dim, TQ_TYPE_UNIFORM_4B, out, size);

// Attention directly on compressed cache — 2.9x faster than FP32
tq_attention(ctx, query, out, n, dim, TQ_TYPE_UNIFORM_4B, scores);
```

---

## Three Breakthroughs

### 1. Faster, Not Just Smaller

Most quantization makes things smaller but slower. TurboQuant makes attention **2.9-4.8x faster than FP32** by computing directly in integer domain — no dequantization needed.

```
FP32:    query × key = dot product       → 22.8 μs
Q4×Q8:   int_query × int_key = int_dot   →  7.8 μs  (2.9x faster)
```

### 2. Real Model Validated

Not synthetic benchmarks — actual [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) KV cache from real inference:

| Type | Compression | Quality | Grade |
|------|-------------|---------|-------|
| **uniform_4b** | 7.5x | cosine 0.994 | **A+** |
| **mixed_4b8** | 6.4x | cosine 0.994 | **A+** |
| uniform_2b | 14.2x | cosine 0.953 | A |

### 3. Community-Proven Architecture

Built on techniques validated by r/LocalLLaMA community and llama.cpp Discussion #20969:

- **Integer dot product** (llama.cpp `vec_dot` pattern)
- **Random Hadamard Transform** (3.9x MSE reduction on Qwen3.5)
- **K/V asymmetric** quantization (Key 4-bit + Value 2-bit = 9.8x)
- **Mixed precision outlier** detection (fp16 outliers + 4-bit base)

---

## How Much Memory Do You Save?

| Model | GPU | FP16 Context | TurboQuant Context | Gain |
|-------|-----|-------------|-------------------|------|
| Qwen3.5-0.8B | 8GB M2 Air | 87K | **286K** | 3.3x |
| Llama-3.2-1B | 16GB RTX 4060 | 445K | **1,462K** | 3.3x |
| Llama-3.2-3B | 24GB RTX 4090 | 164K | **540K** | 3.3x |

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started.md)** | **Build, CLI, Python, C API, llama.cpp — all in one page** |
| [Architecture](docs/architecture.md) | 4-layer design, type system, dispatch |
| [Qwen3.5 Validation](docs/qwen35_validation_results.md) | Real model A/B test results |
| [Integration Guide](docs/integration_guide.md) | llama.cpp, vLLM, Python |
| [llama.cpp Plugin](integrations/llamacpp/README.md) | Step-by-step llama.cpp integration |
| [Format Spec](spec/tq_format_v1.md) | Block structure, bit packing |
| [Changelog](CHANGELOG.md) | Full release notes |

---

## Technical Highlights

- **8 quantization types** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **Integer-domain attention** — Q4×Q8 via ARM `vdotq_s32` / x86 VNNI
- **Zero dependencies** — pure C11/C++17, libc/libm only
- **Thread-safe** — pthread mutex, TSan verified
- **38+ tests** — ASan + UBSan + TSan clean
- **GPU ready** — CUDA + Metal kernels included

---

## References

- **TurboQuant** — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **QJL** — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- **PolarQuant** — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

---

**Developed by [QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
