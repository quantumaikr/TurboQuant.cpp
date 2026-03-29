# TurboQuant — Qwen3.5-0.8B Validation Results

**Date**: 2026-03-29
**Model**: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Total Layers | 24 (hybrid: DeltaNet + Gated Attention) |
| Attention Layers | 6 (every 4th layer) |
| KV Heads | 2 (GQA) |
| Query Heads | 8 |
| Head Dimension | 256 |
| Max Context | 262,144 tokens |

## Quantization Quality

Tested on 4 attention layers × 2 KV heads × 64 tokens:

| Type | BPE | MSE | Attention Cosine | Grade | Compression |
|------|-----|-----|-----------------|-------|-------------|
| **mixed_4b8** | 5.0 | **0.0038** | **0.997** | **A+** | 6.4x |
| **uniform_4b** | 4.2 | 0.0215 | **0.980** | **A** | 7.5x |
| turbo_3b | 7.0 | 0.174 | 0.875 | B | 4.6x |
| uniform_2b | 2.2 | 0.303 | 0.835 | B | 14.2x |
| polar_4b | 4.5 | 0.347 | 0.826 | B | 7.1x |
| qjl_1b | 1.2 | 0.494 | 0.765 | C | 25.6x |

### Key Findings

1. **mixed_4b8 achieves A+ (0.997 cosine)** — best quality on Qwen3.5-0.8B. The 256-dim head allows more effective outlier separation.
2. **uniform_4b remains solid A grade (0.980)** — reliable production choice.
3. **QJL/PolarQuant struggle with 256-dim heads** — designed for 128-dim, quality degrades on larger dimensions.

## RHT (Random Hadamard Transform) Impact

| Method | MSE | Improvement |
|--------|-----|-------------|
| Raw uniform_4b | 0.0215 | baseline |
| **RHT + uniform_4b** | **0.0054** | **3.9x better** |

RHT's effectiveness increases with head_dim (3.5x at dim=128, **3.9x at dim=256**).

## K/V Asymmetric Configuration

| Config | Key Bits | Value Bits | Avg Bits | Compression |
|--------|----------|------------|----------|-------------|
| K4V4 | 4.2 | 4.2 | 4.2 | 7.5x |
| **K4V2** | **4.2** | **2.2** | **3.25** | **9.8x** |
| K2V2 | 2.2 | 2.2 | 2.2 | 14.2x |

**Recommended**: K4V2 (Key 4-bit + Value 2-bit) = 9.8x compression with preserved key quality.

## Memory Impact

| Context | FP16 | Uniform 4-bit | K4V2 | Saved |
|---------|------|---------------|------|-------|
| 4K | 0.05 GB | 0.01 GB | 0.00 GB | 90% |
| 16K | 0.19 GB | 0.05 GB | 0.02 GB | 90% |
| 64K | 0.75 GB | 0.20 GB | 0.08 GB | 90% |
| 128K | 1.50 GB | 0.40 GB | 0.15 GB | 90% |

Note: Qwen3.5-0.8B uses only 6 attention layers (hybrid architecture), so KV cache is smaller than traditional transformers. This amplifies the benefit of quantization on memory-constrained devices.

## Recommendations for Qwen3.5 Models

1. **Best quality**: `mixed_4b8` (A+ at 6.4x compression)
2. **Best balance**: `uniform_4b` (A at 7.5x compression)
3. **Max compression**: K4V2 asymmetric (9.8x with key quality preserved)
4. **Always use RHT**: 3.9x MSE improvement on 256-dim heads

## Reproduction

```bash
# Generate test data
python3 tests/reference/dump_qwen35_kv.py

# Build and run validation
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_BENCH=ON
cmake --build build -j$(nproc)
./build/qwen35_validation
```
