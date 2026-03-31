# TurboQuant.cpp

**Pure C inference engine with [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Release](https://img.shields.io/github/v/release/quantumaikr/TurboQuant.cpp)]()
[![Tests](https://img.shields.io/badge/tests-23%20suites-brightgreen)]()

### Up to 7.1x total K+V compression. Quality preserved.

```
Gemma 3 4B — total K+V memory per token:

  FP16 K+V (llama.cpp):    136.00 KB   (baseline)
  1-bit K + Q4 V:            27.62 KB   (4.9x)   "Paris" ✓  "1+1=2" ✓
  1-bit K + Q2 V:            19.12 KB   (7.1x)   "Paris" ✓  "Mercury, Venus, Earth" ✓
```

Key compression: 10.7x (1-bit sign hash). Value compression: Q4 (3.8x) or Q2 (7.6x). Combined: **up to 7.1x total K+V**.

---

## Why This Matters

LLM attention computes **inner products** `<query, key>`. Standard quantizers minimize reconstruction error (MSE), but introduce **systematic bias** in inner product estimation.

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) proved this gap and showed how to close it:

- **Keys**: RHT + Lloyd-Max codebook + QJL residual → **unbiased** inner product estimation at any bit-width
- **Values**: RHT + Lloyd-Max codebook → **MSE-optimal** reconstruction for weighted sum

We implemented both in pure C, and pushed keys to **1 bit** — attention via XOR + popcount.

---

## Compression Options

```bash
# Key compression (affects attention scoring)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b       # 1-bit keys (10.7x)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b       # 3-bit keys (4.6x)

# Value compression (affects output reconstruction)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4  # + Q4 values → 4.9x total
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2  # + Q2 values → 7.1x total

# Memory stats
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4 -M
```

### Total K+V Compression Table

| Config | K bits | V bits | K+V/token | Total compression | Quality |
|--------|--------|--------|-----------|-------------------|---------|
| FP16 (baseline) | 16 | 16 | 136.00 KB | 1.0x | reference |
| uniform_4b + FP16 V | 4 | 16 | 86.06 KB | 1.6x | baseline |
| 1-bit K + FP16 V | 1 | 16 | 74.38 KB | 1.8x | greedy identical up to ~120 tok |
| **1-bit K + Q4 V** | **1** | **4** | **27.62 KB** | **4.9x** | **"Paris" ✓ "1+1=2" ✓** |
| **1-bit K + Q2 V** | **1** | **2** | **19.12 KB** | **7.1x** | **"Paris" ✓ planets ✓** |

### Memory at 32K Context (Gemma 3 4B)

```
FP16 K+V:              4,352 MB
1-bit K + Q4 V:           885 MB   (4.9x, 3.4 GB saved)
1-bit K + Q2 V:           613 MB   (7.1x, 3.7 GB saved)
```

> **Note on quality:** With K-only quantization (V as FP16/FP32), greedy decode is byte-identical
> up to ~120 tokens. With V quantization (Q4/Q2), outputs diverge earlier but remain coherent
> and factually correct. This is expected — V quantization affects reconstruction directly.

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

---

## The Algorithm

```
Keys (attention scoring — needs unbiased inner products):
  key → normalize → RHT → Lloyd-Max codebook (b-1 bits) → QJL signs (1 bit)
  1-bit extreme: skip codebook, store signs only → XOR + popcount attention

Values (weighted sum — needs MSE-optimal reconstruction):
  value → Q4 or Q2 per-block quantization → dequantize on the fly during output
```

| Component | For Keys | For Values |
|-----------|----------|------------|
| **Goal** | Unbiased inner product | Low MSE reconstruction |
| **Method** | RHT + codebook + QJL | Per-block scale + quantize |
| **1-bit** | Sign hash (XOR+popcount) | Not recommended |
| **Best config** | 1-bit (10.7x key compression) | Q4 (3.8x value compression) |

---

## Supported Models

| Model | Params | Speed (Q4, 6T) | Verified |
|-------|--------|----------------|----------|
| **Gemma 3 4B** | 4B | 20.2 tok/s | "Paris" ✓, planets ✓ |
| **Qwen3.5-0.8B** | 752M | 80.1 tok/s | 0.999 cosine vs PyTorch |
| **Gemma 3 270M** | 270M | 176 tok/s | per-layer exact match |

Multi-architecture: Qwen3.5 (DeltaNet hybrid) + Gemma 3 (sliding window). Gemma 4 ready.

---

## Under the Hood

- **10,000+ lines of pure C** — zero external dependencies
- **11 quantization types** — Uniform, Mixed, PolarQuant, QJL, TurboQuant KV (1/3/4-bit)
- **K+V independent compression** — 1-bit keys (XOR+popcount) + Q4/Q2 values
- **Faithful ICLR 2026 implementation** — RHT + Lloyd-Max + QJL residual
- **Multi-architecture** — Qwen3.5 (DeltaNet) + Gemma 3 (sliding window + GeGLU)
- **NEON vectorized** — matmul, attention, Hamming distance, FP16 conversion
- **23 test suites** — KV roundtrip, attention accuracy, codebook, Q2 weights

---

## The Journey

```
Day 1:  Empty → KV compression library → inference engine → 82 tok/s
Day 2:  Gemma 3 (270M+4B) → TurboQuant algorithm → 1-bit K → V quantization

Lines of C:      10,000+
Test suites:     23
Models:          3 (Gemma 4B, Qwen 0.8B, Gemma 270M)
Best K+V:        7.1x total compression (1-bit K + Q2 V)
32K savings:     3.7 GB vs FP16
```

---

## References

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — Online Vector Quantization with Near-optimal Distortion Rate
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit Quantized JL Transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar Coordinate Quantization

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
