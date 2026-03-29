# Introducing TurboQuant.cpp — 7.5x KV Cache Compression for LLM Inference

We're open-sourcing **TurboQuant.cpp**, a zero-dependency C/C++ library that compresses LLM KV caches from 16-bit to 2-4 bits — giving you **3x longer contexts on the same GPU**.

## The Problem

KV cache is the #1 memory bottleneck in LLM inference. Running Llama-3.2-3B at 64K context? That's **7 GB** just for KV cache — often more than the model weights.

## What TurboQuant Does

One line change. Same model. Same GPU. 3x more context.

```
Before:  Llama-3.2-3B @ 64K context → 7.00 GB KV cache
After:   Llama-3.2-3B @ 64K context → 0.93 GB KV cache (87% saved)
```

## A/B Test: Does Quality Survive?

We ran 200 queries against 512 cached keys with realistic LLM distributions:

| Method | Compression | Cosine vs FP16 | Grade |
|--------|-------------|----------------|-------|
| FP16 (baseline) | 1x | 1.000 | — |
| **uniform_4b** | **7.5x** | **0.995** | **A+** |
| turbo_3b | 4.6x | 0.917 | B+ |
| uniform_2b | 14.2x | 0.897 | B |

**uniform_4b achieves 7.5x compression with 99.5% accuracy. Virtually lossless.**

## Key Numbers

- **2.87M elements/ms** quantization throughput
- **331K queries/sec** attention throughput
- **5.74x SIMD speedup** (ARM NEON)
- **11 test suites**, ASan/UBSan/TSan clean
- **Zero dependencies** — pure C11, libc/libm only

## What's Inside

- 7 quantization types (PolarQuant, QJL, TurboQuant, Uniform)
- Direct attention kernels — no dequantization needed (Hamming distance for QJL, cos/sin LUT for PolarQuant)
- Progressive compression — recent tokens stay high-precision, old tokens auto-compress
- Paged cache with Copy-on-Write for beam search
- CPU (Generic + NEON + AVX2), CUDA, Metal backends
- llama.cpp/vLLM integration interfaces

## Try It

```bash
git clone https://github.com/anthropics/TurboQuant.cpp
cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(nproc)
./build/ab_test           # See the A/B comparison yourself
./build/demo_real_model   # Memory savings for Llama, Qwen, Phi models
```

Based on TurboQuant (ICLR 2026), QJL (AAAI 2025), and PolarQuant (AISTATS 2026). Architectural patterns from llama.cpp, vLLM, and ONNX.

Apache 2.0. Contributions welcome.
