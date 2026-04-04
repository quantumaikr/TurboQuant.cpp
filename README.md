# quant.cpp

![quant.cpp Hero](docs/assets/hero.png)

Embeddable LLM inference in pure C. Also ships as [**quant.h**](#single-header-mode) — a single-header library.

33K LOC. No external libraries. Read it in an afternoon.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/quant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-34%20pass-brightgreen)]()

---

## What quant.cpp does

**~4x longer context on the same hardware.** KV cache compression reduces per-token memory by 3.8x, extending context proportionally.

| Hardware | Model | FP16 KV | Compressed KV | Gain |
|----------|-------|---------|---------------|------|
| 8GB Laptop | Llama 8B (Q4) | ~16K tokens | ~61K tokens | 3.8x |
| 16GB Mac Air | SmolLM2 1.7B | ~78K tokens | ~298K tokens | 3.8x |
| **16GB Mac** | **Gemma 4 26B-A4B** | **~8K tokens** | **~20K tokens** | **3.5x** |
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
#include <stdio.h>

static void on_token(const char* text, void* ud) {
    (void)ud;
    printf("%s", text);
    fflush(stdout);
}

int main() {
    quant_model* m = quant_load("model.gguf");
    quant_ctx*   c = quant_new(m, NULL);
    quant_generate(c, "Hello!", on_token, NULL);
    quant_free_ctx(c);
    quant_free_model(m);
}
```

```bash
cc app.c -o app -lm -lpthread    # that's it
```

15K lines, 628KB. No cmake, no build system, no framework.

**Full API** (6 functions):

| Function | Description |
|----------|-------------|
| `quant_load(path)` | Load a GGUF model file |
| `quant_new(model, config)` | Create inference context (config=NULL for defaults) |
| `quant_generate(ctx, prompt, callback, userdata)` | Stream tokens via callback |
| `quant_ask(ctx, prompt)` | Generate and return full string (caller frees) |
| `quant_free_ctx(ctx)` | Free context |
| `quant_free_model(model)` | Free model |

**Config options:**

```c
quant_config cfg = {
    .temperature = 0.7f,    // sampling temperature
    .top_p       = 0.9f,    // nucleus sampling
    .max_tokens  = 256,     // generation limit
    .n_threads   = 4,       // matmul threads
    .kv_compress = 1,       // 0=off, 1=4-bit K+V, 2=delta+3-bit
};
quant_ctx* c = quant_new(model, &cfg);
```

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

### QK-norm aware compression (Gemma 4)

Models with QK-norm (Gemma 4) normalize key vectors to the unit sphere, creating extremely sparse distributions (256 dimensions, only ~56 active). Standard 4-bit quantization destroys directional information (cosine similarity drops to 0.62).

quant.cpp automatically detects QK-normed models and stores keys in FP32 while quantizing only values to Q4. This preserves perfect key precision with **3.5x V memory reduction**.

| Config | Compression | Quality (Gemma 4) |
|--------|-------------|-------------------|
| FP32 K + Q4 V (auto) | 3.5x V savings | Correct: "Paris", "서울" |
| 4-bit K (forced) | 3.8x total | Broken: cosine=0.62 |

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
| **Llama 3.2 3B-Instruct** | **Llama 3** | **3B** | **Verified (11.6 tok/s)** |
| **Gemma 4 26B-A4B-it** | **Gemma 4 MoE** | **26B (4B active)** | **Verified** |

### Gemma 4 26B-A4B (NEW)

Full support for Gemma 4's hybrid MoE architecture:

- **Dual-FFN**: parallel Dense MLP + 128-expert MoE per layer
- **Hybrid attention**: 25 sliding (head_dim=256) + 5 full (head_dim=512) layers
- **QK-norm aware KV compression**: auto FP32 keys + Q4 values (3.5x savings)
- **Learned RoPE** with per-layer frequency factors
- **IQ3_XXS/IQ4_NL** fused dot with NEON optimization for MoE experts
- **GeGLU** activation (NEON-accelerated fast tanh approximation)

```bash
# Gemma 4 26B inference with KV compression
./build/quant gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  -p "<start_of_turn>user\nWhat is the capital of France?\n<end_of_turn>\n<start_of_turn>model\n" \
  -n 50 -j 8 -T 0.0 -k uniform_4b -v q4
# Output: "The capital of France is **Paris**."
```

Architectures: Llama/Qwen3.5 (shared path), Gemma 3/4 (sliding + full attention), Qwen2-MoE, Gemma 4 MoE (dual-FFN + hybrid attention).

GGUF format. Load any llama.cpp-compatible model file.

---

## Backends

| Backend | Platform | Status |
|---------|----------|--------|
| NEON | ARM CPU | Production (5.8x SIMD speedup) |
| AVX2 | x86 CPU | Production |
| Metal | Apple Silicon | Verified |
| CUDA | NVIDIA GPU | Compiles |
| Vulkan | Cross-platform | Compiles |

### Performance (Gemma 4 26B-A4B on M1 Pro, 8 threads)

| Component | Time/token | Notes |
|-----------|-----------|-------|
| Attention matmul (Q8_0) | 168ms | NEON two-accumulator fused dot |
| MoE experts (IQ3_XXS/IQ4_NL) | 72ms | NEON fused dot + GeGLU NEON |
| Output projection (Q6_K) | included | GGUF on-the-fly fused dot |
| **Total** | **257ms** | **3.9 tok/s** |

---

## FAQ

**How is this different from llama.cpp?**

llama.cpp is a full-featured inference framework (250K+ LOC). quant.cpp is a minimal engine (33K LOC) you can read, modify, and embed in your own C/C++ project.

**llama.cpp/ollama already has Q4 KV quantization. How is yours better?**

Both use 4 bits per element, but quality differs significantly. On SmolLM2 1.7B:
- llama.cpp Q4_0 KV: PPL **+10.6%** (noticeable degradation)
- quant.cpp 4-bit K: PPL **+0.0%** (lossless)

The difference: llama.cpp applies the same quantization scheme to both K and V. quant.cpp quantizes K and V independently with type-appropriate methods. Additionally, quant.cpp offers delta compression — encoding the difference between adjacent keys instead of absolute values — which pushes to 3-bit at only +1.3% PPL. llama.cpp has no equivalent.

**Can I embed this in my app?**

Yes. Two options:
1. **Single-header** (easiest): Copy `quant.h` into your project. `#define QUANT_IMPLEMENTATION` in one .c file. Done.
2. **Full library**: Link against `libturboquant.a` and call `tq_load_model()` / `tq_generate()`.

Works on Linux, macOS, Windows, iOS, Android, and WASM. Thread pool is global but mutex-protected.

**What's the difference between quant.h and the full build?**

`quant.h` is the core inference engine (15K LOC) in a single file. The full build (33K LOC) adds GPU backends (Metal, CUDA, Vulkan), MoE routing, advanced quantization types, CLI tools, and benchmarks. Use quant.h for embedding; use the full build for research and development.

**15K lines in a header — isn't that too big?**

stb_image.h is 7.8K lines. sqlite3.c (the amalgamation) is 240K lines. quant.h sits in between at 15K — large for a header, small for an inference engine. Compile time is ~1.7 seconds on Apple M3. Binary size is 254KB. If compile time is a concern, use the full CMake build and link against `libturboquant.a` instead.

**How does this compare to Karpathy's llm.c?**

Similar philosophy: minimal C, educational, no dependencies. Key differences: quant.h supports quantized weight formats (Q4_K_M, Q8_0, IQ2) and multiple architectures (Llama, Qwen, Gemma) via the GGUF loader. llm.c targets a single model with FP32 weights. quant.h also includes KV cache compression. Think of llm.c as the textbook and quant.h as the production-ready version of the same idea.

**No GPU — is this useless?**

Depends on your use case. If you need 100+ tok/s on large models, use llama.cpp with Metal/CUDA. If you need to embed inference in an iOS app, a WASM module, a game engine, or an IoT device where there is no GPU API — quant.h works. CPU inference on Apple Silicon gets 25 tok/s on a 1.7B model, which is fine for assistants, autocomplete, and background processing.

**Does it work on Windows?**

Yes. The header includes `#ifdef _WIN32` guards for mmap (uses `CreateFileMapping`/`MapViewOfFile`), threading, and file I/O. Compile with MSVC or MinGW: `cl app.c /O2` or `gcc app.c -o app -lm -lpthread`.

**How do I get a GGUF model file?**

Download any GGUF from [Hugging Face](https://huggingface.co/models?library=gguf). Recommended starter model: [SmolLM2-1.7B-Instruct-Q8_0](https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF) (1.8GB). No conversion needed — GGUF files work directly.

**Is this AI-generated code?**

Developed with Claude Code as an AI-assisted development tool, same way others use Copilot. The architecture, algorithm choices, bug fixes, and every PPL measurement are human-directed and verified. The code is 33K lines of C — feel free to read it.

**What about sub-3-bit quantization?**

Tested extensively: 2-bit delta, sub-block scaling, multi-hash, error feedback, NF2, online SVD. None reached acceptable quality. The barrier: per-step cosine 0.997 compounds to 0.885 after 200 steps. 3-bit + delta is the practical minimum.

**Can it run in the browser (WASM)?**

The code is pure C11 with no platform-specific dependencies in the core path. Emscripten compilation is supported. A browser demo with a small model is on the roadmap.

---

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression theory
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — Quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization

---

**[QuantumAI](https://quantumai.kr)** | [GitHub](https://github.com/quantumaikr/quant.cpp)

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date)](https://star-history.com/#quantumaikr/quant.cpp&Date)
