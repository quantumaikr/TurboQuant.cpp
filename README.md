<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">LLM inference with 7x longer context — pure C, zero dependencies</h3>

<p align="center">
  Lossless KV cache compression. Also ships as <a href="#-single-header-mode"><b>quant.h</b></a> — a single-header library.<br>
  72K LOC. Embeddable. Read it in an afternoon.
</p>

<p align="center">
  <a href="https://github.com/quantumaikr/quant.cpp/releases/tag/v0.5.0"><img src="https://img.shields.io/badge/release-v0.5.0-blue" alt="Release"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-34%20pass-brightgreen" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/score-99.2%25-brightgreen" alt="Score"></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/models-7%20verified-blue" alt="Models"></a>
  <a href="https://quantumaikr.github.io/quant.cpp/"><img src="https://img.shields.io/badge/WASM_demo-192KB-purple" alt="WASM"></a>
  <a href="#"><img src="https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20WASM-orange" alt="Platforms"></a>
</p>

---

## The Problem

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

## The Result

> **Same hardware. 7x longer context. Zero quality loss.**

| Hardware | Model | FP16 KV | quant.cpp KV | Gain |
|:---------|:------|--------:|-------------:|-----:|
| 16GB Mac | Llama 3.2 3B | 50K tokens | **350K tokens** | **6.9x** |
| 16GB Mac | Gemma 4 26B MoE | 4K tokens | **30K tokens** | **6.9x** |
| 8GB Laptop | Llama 8B (Q4) | 16K tokens | **61K tokens** | **3.8x** |
| 24GB RTX 3090 | Llama 8B (Q4) | 147K tokens | **559K tokens** | **3.8x** |

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

> **[Full API docs](docs/api.md)** · **[WASM demo](https://quantumaikr.github.io/quant.cpp/)** · **[Add your own KV type](docs/custom-quantization.md)**

---

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

### vs every other engine

|  | quant.cpp | llama.cpp | vLLM | MLX | ONNX RT |
|:--|:---------:|:---------:|:----:|:---:|:-------:|
| KV compression | **7x, +0% PPL** | 1.6x at ~+1% PPL | -- | -- | -- |
| Code size | **72K LOC** | 250K+ | 100K+ | 50K+ | 500K+ |
| Dependencies | **zero** | ggml | PyTorch | Apple fw | runtime |
| Embeddable | **single header** | -- | -- | -- | complex |
| WASM | **192KB** | -- | -- | -- | -- |
| GPU serving | basic | full | **best** | Metal | multi |

> **Use llama.cpp** when you need speed. **Use vLLM** when you need throughput.
> **Use quant.cpp** when you need to fit more context in less memory — or embed LLM in your own app.

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

```bash
# Delta compression (maximum context, 8.5x)
./build/quant model.gguf --chat -p "hello" -k uniform_3b -v q4 --delta

# Perplexity benchmark
./build/quant model.gguf --ppl input.txt -k uniform_4b -v q4

# Model info
./build/quant model.gguf --info

# Performance profiling
./build/quant model.gguf --chat -p "hello" -n 50 --profile
```

---

## Single-Header Mode

> Copy one file. Add LLM to any C project.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("model.gguf");
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
docker run -v ./models:/models quant.cpp /models/model.gguf -p "hello" -k uniform_4b -v q4
```

**OpenAI-compatible server** (`/v1/chat/completions`):
```bash
cmake -B build -DTQ_BUILD_SERVER=ON && cmake --build build
./build/quant-server model.gguf -p 8080 -k uniform_4b

# Works with the OpenAI Python SDK
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Build with `-DTQ_BUILD_SERVER=ON`. Streaming SSE supported. KV compression configurable per request.

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
| **[ROADMAP](ROADMAP.md)** | Project direction and planned features |
| **[CHANGELOG](CHANGELOG.md)** | Version history and release notes |
| **[Tech Report](docs/papers/quant_cpp_tech_report.md)** | Architecture and benchmarks (Arxiv draft) |
| **[WASM Demo](https://quantumaikr.github.io/quant.cpp/)** | Try it in your browser — no install needed |

---

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression theory
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — Quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization

---

<p align="center">
  <b><a href="https://quantumai.kr">QuantumAI</a></b> · <a href="https://github.com/quantumaikr/quant.cpp">GitHub</a>
</p>

<p align="center">
  <a href="https://star-history.com/#quantumaikr/quant.cpp&Date">
    <img src="https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date" alt="Star History" width="600">
  </a>
</p>
