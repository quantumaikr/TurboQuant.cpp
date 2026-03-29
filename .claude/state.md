# TurboQuant.cpp — Session State

**Last updated**: 2026-03-29 (v0.9 Q4 weights — 38 tok/s)
**Last commit**: 4415bcb

## Speed Progression
```
PyTorch CPU:        0.8 tok/s
v0.8 FP32:          5   tok/s  (6x PyTorch)
v0.8 Q8+threads:   21   tok/s  (26x)
v0.9 Q4+threads:   38   tok/s  (48x) ← current
llama.cpp Q4_K_M:  ~50   tok/s  ← target
```

## What Works
- ✅ 38.2 tok/s CPU (Q4 weights, 4 threads, Qwen3.5-0.8B)
- ✅ Q4 weights: 270 MB, Q8: 533 MB (vs 2.1 GB FP32)
- ✅ Self-contained C inference engine, 0 dependencies
- ✅ DeltaNet + Self-Attention hybrid forward pass
- ✅ KV cache quantization (Q4, 7.5x compression)
- ✅ Integer Q4×Q8 attention
- ✅ 19 C++ + 22 Python tests

## What Needs Work
1. Close llama.cpp gap: 38 → 50 tok/s (matmul tiling)
2. Q4 quality on short prompts
3. Metal GPU inference
4. More model architectures
