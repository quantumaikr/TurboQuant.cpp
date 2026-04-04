# quant.cpp

![quant.cpp Hero](docs/assets/hero.png)

Embeddable LLM inference in pure C.

33K LOC. No external libraries. Read it in an afternoon.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/quant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-34%20pass-brightgreen)]()

---

## What quant.cpp does

**~4x longer context on the same hardware.** KV cache compression reduces per-token memory by 3.8x, extending context proportionally.

| Hardware | Model | FP16 KV | 4-bit K + Q4 V | Gain |
|----------|-------|---------|----------------|------|
| 8GB Laptop | Llama 8B (Q4) | ~16K tokens | ~61K tokens | 3.8x |
| 16GB Mac Air | SmolLM2 1.7B | ~78K tokens | ~298K tokens | 3.8x |
| 24GB RTX 3090 | Llama 8B (Q4) | ~147K tokens | ~559K tokens | 3.8x |

*Estimates based on KV memory reduction. Actual context depends on available memory after model weights.*

```bash
./quant model.gguf -p "hello"
```

---

## Why quant.cpp

|  | quant.cpp | llama.cpp |
|--|-----------|-----------|
| Code | **33K LOC**, pure C | 250K+ LOC, C++ |
| Design | Read, modify, embed | Feature-complete |
| Dependencies | libc + pthreads only | ggml framework |
| KV compression | 4-bit: PPL **+0.0%**, 3-bit: +1.3% | PPL +10.6% |

quant.cpp is not a fork. It's a standalone engine built from scratch for one goal: **LLM inference you can understand, customize, and ship inside your own product.**

- **Read** — 33K lines. The full forward pass fits in one file. You can trace every computation.
- **Modify** — Pure C11, modular. Add your own quantization type, swap the attention kernel, change the sampling strategy.
- **Embed** — No frameworks, no package managers. Copy the source into your project. Compiles on any platform with a C compiler.

---

## Single-Header Mode

Copy one file. Add LLM to any C project.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("model.gguf");
    quant_ctx*   c = quant_new(m, NULL);
    quant_generate(c, "Hello!", print_token, NULL);
    quant_free_ctx(c);
    quant_free_model(m);
}
```

```bash
cc app.c -o app -lm -lpthread    # that's it
```

15K lines, 628KB. No cmake, no build system, no dependencies.

---

## Quick Start (Full Build)

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run inference with a GGUF model
./build/quant model.gguf -p "hello"

# KV compression: 4-bit keys + Q4 values (3.8x, recommended)
./build/quant model.gguf -p "hello" -k uniform_4b -v q4

# Delta compression: 3-bit keys + Q4 values (4.3x, best compression)
./build/quant model.gguf -p "hello" -k uniform_3b -v q4 --delta

# Measure perplexity
./build/quant model.gguf --ppl input.txt -k uniform_4b -v q4
```

---

## KV Cache Compression

### Modes

| Config | Compression | PPL vs FP32 (WikiText-2) | When to use |
|--------|-------------|--------------------------|-------------|
| delta + 3b K + Q4 V | ~4.3x | **+1.3%** | Maximum context length |
| delta + 4b K + Q4 V | ~3.8x | ~0% | Maximum quality |
| uniform 4b K + Q4 V | 3.8x | -7.8% | Simple, no delta overhead |
| uniform 4b K + FP16 V | 1.6x | +0.0% | Lossless baseline |

### Delta compression

Standard KV caching stores each key vector as-is. Delta mode stores `key[t] - reconstruct(key[t-1])` — like video P-frames.

Adjacent keys differ by ~30% of their absolute range. This smaller range means 3-bit quantization works. Without delta, 3-bit gives PPL +62%. With delta: **+1.3%**.

Every 64 tokens, an FP32 I-frame is stored to prevent drift.

### WikiText-2 PPL (SmolLM2 1.7B, standard benchmark)

| Config | PPL | vs FP32 |
|--------|-----|---------|
| FP32 baseline | 14.63 | -- |
| uniform 4b K + FP16 V | 14.63 | **+0.00%** |
| uniform 4b K + Q4 V | 14.57 | -0.4% |
| delta + 3b K + Q4 V | 14.82 | +1.3% |
| uniform 3b (no delta) | — | +62% |

Cross-model (4b K + Q4 V): SmolLM2 1.7B (-1.6%), Qwen3.5 0.8B (+0.9%), Qwen3.5 4B (+0.6%).

---

## Supported Models

| Model | Architecture | Params | Status |
|-------|-------------|--------|--------|
| SmolLM2-1.7B | Llama | 1.7B | PPL verified |
| Qwen3.5-0.8B | Qwen3.5 (DeltaNet) | 752M | PPL verified |
| Qwen3.5-4B | Qwen3.5 (DeltaNet) | 4B | PPL verified |
| Qwen3.5-35B-A3B | Qwen2-MoE | 35B (3B active) | Working |
| Gemma 3 270M | Gemma 3 | 270M | Working |
| Gemma 4 E2B | Gemma 4 | 2B | Experimental (non-standard GGUF) |

Architectures: Llama/Qwen3.5 (shared path), Gemma 3/4 (sliding + full attention), Qwen2-MoE.

GGUF format. Load any llama.cpp-compatible model file.

---

## Backends

| Backend | Platform | Status |
|---------|----------|--------|
| NEON | ARM CPU | Production |
| AVX2 | x86 CPU | Production |
| Metal | Apple Silicon | Verified |
| CUDA | NVIDIA GPU | Compiles |
| Vulkan | Cross-platform | Compiles |

---

## FAQ

**How is this different from llama.cpp?**

llama.cpp is a full-featured inference framework (250K+ LOC). quant.cpp is a minimal engine (33K LOC) you can read, modify, and embed in your own C/C++ project. On KV compression specifically: llama.cpp Q4_0 gives PPL +10.6% on SmolLM2 1.7B; quant.cpp gives +0.0%.

**Can I embed this in my app?**

Yes. Pure C11, zero dependencies. Copy the source files, link against libc/libm, and call `tq_load_model()` / `tq_generate()`. Works on Linux, macOS, Windows, iOS, Android, and WASM. Thread pool is global but mutex-protected.

**What about sub-3-bit quantization?**

Tested extensively: 2-bit delta, sub-block scaling, multi-hash, error feedback, NF2, online SVD. None reached acceptable quality. The barrier: per-step cosine 0.997 compounds to 0.885 after 200 steps. 3-bit + delta is the practical minimum.

---

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression theory
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — Quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization

---

**[QuantumAI](https://quantumai.kr)** | [GitHub](https://github.com/quantumaikr/quant.cpp)

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date)](https://star-history.com/#quantumaikr/quant.cpp&Date)
