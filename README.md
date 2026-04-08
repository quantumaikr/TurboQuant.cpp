<p align="center">
  <img src="docs/assets/hero.png" alt="quant.cpp" width="600">
</p>

<h3 align="center">The single-header C engine for KV-compressed LLM inference</h3>

<p align="center">
  Production: <code>uniform_4b</code> KV cache (4–7x compression at +6% PPL on Llama 3.2 3B).<br>
  Research: building blocks for <a href="https://arxiv.org/abs/2504.19874">TurboQuant</a>, <a href="https://arxiv.org/abs/2502.02617">PolarQuant</a>, <a href="https://arxiv.org/abs/2406.03482">QJL</a> — 7 KV quantization types in one engine.<br>
  72K LOC pure C, zero dependencies. Ships as <a href="#-single-header-mode"><b>quant.h</b></a> — drop one file into any project.<br>
  Runs everywhere a C compiler does: <b>iOS · Android · WASM · MSVC · microcontrollers</b>.
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

> **Same hardware. 4–7x longer context. PPL measured and disclosed.**

### Llama 3.2 3B Instruct — PPL × Speed (FP32 KV NEON = 13.56 PPL @ 14.8 tok/s)

> 9 rounds of Karpathy iteration closed the quant-KV speed gap to FP32 KV from **−45% to −8%**, while delivering 5.8–7.1× memory compression. We do not (yet) beat fp32 in raw speed, but we get within 8% of it for ~7× less memory.

| KV Config | Bytes/block | Compression | PPL | Δ vs FP32 | tok/s | vs FP32 speed |
|:----------|------------:|------------:|----:|----------:|------:|--------------:|
| FP32 reference (NEON) | — | 1× | 13.56 | — | 14.83 | baseline |
| **`turbo_kv_5b`** 🏆 quality | 88 | 5.8× | **13.65** | **+0.7%** | **13.13** | **−11.5%** |
| `turbo_kv_4bo` 🧪 | 96 | 5.3× | 13.90 | +2.5% | 12.7 | −14% |
| **`turbo_kv_4b`** ⭐ default | **72** | **7.1×** | **14.33** | **+5.7%** | **13.67** | **−7.8%** |
| `turbo_kv_3b` | 56 | 9.1× | 15.36 | +13.3% | 13.4 | −9.6% |
| `turbo_kv_3bo` 🧪 | 80 | 6.4× | 14.17 | +4.5% | 9.3 | −37% |
| `uniform_4b` | 68 | 7.5× | 14.60 | +7.7% | 11.7 | −21% |
| llama.cpp `q4_0` KV (lit.) | ~70 | ~7.3× | ~14.99 | +10.6% | — | — |

```
                  PPL Degradation vs FP32           Speed vs FP32 KV
                       (lower is better)              (higher is better)

  turbo_kv_5b     │█ +0.7%                       █████████ −11.5%
  turbo_kv_4bo    │██▌ +2.5%                     ████████ −14%
  turbo_kv_4b ⭐  │█████ +5.7%                   ██████████ −7.8%
  turbo_kv_3b     │█████████████ +13.3%          █████████ −9.6%
  uniform_4b      │██████ +7.7%                  ███████ −21%
  llama.cpp q4_0  │██████████ +10.6%             — (not measured)
  FP32 reference  │ ← 0%                         14.83 tok/s ←
                   0%   +5%   +10%               0   25%   50%   75%   100%
```

`turbo_kv_4b` (default) and `turbo_kv_5b` (quality) are the Pareto-optimal recommendations: **5.8–7.1× memory compression at 92% of FP32 KV speed.** Full Karpathy-loop history (9 rounds across 3 sessions) in [bench/results/turboquant_reproduction.md](bench/results/turboquant_reproduction.md).

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

### vs production inference engines

|  | quant.cpp | llama.cpp | vLLM | MLX |
|:--|:---------:|:---------:|:----:|:---:|
| KV quantization | **TurboQuant + 6 schemes** | Q8_0/Q5_0 (2x) | -- | -- |
| Code size | **72K LOC** | 250K+ | 100K+ | 50K+ |
| Embeddable | **single header** | library | library | framework |
| Read in an afternoon | **✅** | ❌ | ❌ | ❌ |
| GPU throughput | basic | full | **best** | Metal |

> **Use llama.cpp** for speed on a workstation. **Use vLLM** for batch serving.
> **Use quant.cpp** when you need to ship LLM inference inside something — an app, a game, a website, a device.

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

## Python

```bash
cd bindings/python && pip install .
```

```python
from quantcpp import Model

with Model("model.gguf", kv_compress=1) as m:
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

- **HIGGS** — Malinovskii, Panferov, Ilin, Guo, Richtárik, Alistarh. *Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*. Nov 2024. [arXiv:2411.17525](https://arxiv.org/abs/2411.17525). HIGGS introduced the **Random Hadamard Transform + MSE-optimal grid quantization** pattern (for weight quantization). Our `tq_rht.c` Walsh-Hadamard + Rademacher implementation follows this pattern. *Credit to Tim Dettmers ([discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969#discussioncomment-16481725)) for pointing this out.*
- **TurboQuant** — Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). TurboQuant applies the rotation pattern to **KV cache** with a 1-bit QJL residual stage and per-channel outlier handling. Our work started as a literal port of TurboQuant; through 9 rounds of Karpathy-loop iteration we simplified it (dropped QJL, dropped outlier channels) into the current Variant F. We do not claim our shipped variant is the TurboQuant algorithm — it is an empirically-derived simplification.
- **PolarQuant** — *Quantizing KV Caches with Polar Transformation*. AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617). The polar-coordinate KV quantization that our `tq_polar.c` baseline implements.
- **QJL** — *Quantized Johnson-Lindenstrauss Transform for KV Cache Compression*. AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482). The 1-bit sketch building block. Used in our `tq_qjl.c` baseline; we found it contributed ~zero to attention scores in the Variant F regime and dropped it.
- [Google Research blog post on TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

**Honest attribution**: Variant F's structure (RHT + scalar grid quantization) is closest to HIGGS in spirit, applied to KV cache like TurboQuant, with both the QJL residual and the outlier channel split removed through ablation. If you use quant.cpp in academic work, please cite all three (HIGGS, TurboQuant, PolarQuant) and this repository.

---

<p align="center">
  <b><a href="https://quantumai.kr">QuantumAI</a></b> · <a href="https://github.com/quantumaikr/quant.cpp">GitHub</a>
</p>

<p align="center">
  <a href="https://star-history.com/#quantumaikr/quant.cpp&Date">
    <img src="https://api.star-history.com/svg?repos=quantumaikr/quant.cpp&type=Date" alt="Star History" width="600">
  </a>
</p>
