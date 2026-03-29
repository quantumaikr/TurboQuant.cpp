# TurboQuant.cpp — Session State

**Last updated**: 2026-03-29 (grow round 1)
**Last commit**: pending
**Score**: 99.7%

## Current Status

### What Works
- ✅ Self-contained inference engine (0 dependencies, pure C)
- ✅ Multi-threaded matmul (4 threads: 31 tok/s inference, 1.56x speedup)
- ✅ Qwen3.5-0.8B: loads, tokenizes, generates correct text
- ✅ DeltaNet + Self-Attention hybrid forward pass (layer-by-layer validated)
- ✅ KV cache quantization library (8 types, integer Q4×Q8 attention)
- ✅ 19 C++ test suites, 22 Python tests
- ✅ CLI tools: tq_run (-j threads), tq, tq_chat, tq_realtime_demo

### What Needs Work (Priority Order)
1. **KV cache in inference**: tq_forward stores keys in FP32, not TurboQuant quantized
2. **Memory**: 3.3GB for BF16→FP32 conversion (should stream/quantize weights)
3. **Weight quantization**: Q8/Q4 weights for 2x memory reduction
4. **Metal GPU inference**: Apple GPU for matmul
5. **tok/s display**: show generation speed in tq_run output

### Key Metrics
| Metric | Value |
|--------|-------|
| CPU inference (4 threads) | ~31 tok/s (Qwen3.5-0.8B, excl. loading) |
| CPU inference (1 thread) | 12.8 tok/s |
| PyTorch CPU | 0.8 tok/s (16-39x slower) |
| PyTorch MPS | 10 tok/s (3x slower than our CPU) |
| KV compression | 7.5x (uniform_4b) |
| Integer attention | 2.9-4.8x faster than FP32 |
| Real model cosine | 0.994 (A+) |
| Tests | 19 C++ + 22 Python |

### Files to Read First
- `.claude/state.md` — THIS FILE (session state)
- `program.md` — Agent task specification
- `CLAUDE.md` — Project guide + methodology
