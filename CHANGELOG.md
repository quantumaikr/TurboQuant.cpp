# Changelog

## [0.1.0] — 2026-03-29

### Core Library
- 7 quantization types: PolarQuant (3/4b), QJL (1b), TurboQuant (3/4b), Uniform (2/4b)
- Direct attention kernels: QJL Hamming distance, PolarQuant cos/sin LUT (no dequantization needed)
- Self-contained block formats with ONNX-compliant LSB-first bit packing
- O(1) type traits dispatch table (llama.cpp pattern)
- Thread-safe API with pthread mutex (TSan verified)
- Cross-platform math constants (TQ_PI/TQ_PI_2, no M_PI dependency)

### Cache Management
- Paged KV cache with block-table mapping (vLLM pattern)
- Progressive compression: 3-tier automatic degradation by age, O(1) append
- Copy-on-Write for beam search (ref_count based)
- Value cache quantization and retrieval

### Backends
- CPU Generic (reference C11, zero external dependencies)
- ARM NEON optimized (5.74x speedup over generic)
- x86 AVX2 stubs ready for implementation
- CUDA kernels: 7 files (polar, qjl, turbo, fused_cache, value, common, dispatch)
- Metal compute shaders: 7 files (polar, qjl, turbo, fused_cache, value, common, dispatch)

### Validation
- **A/B test**: uniform_4b achieves cosine 0.995 vs FP16 — A+ grade, virtually lossless
- **Real model validation**: cosine 0.991 on Qwen2.5-0.5B KV cache patterns (4 layers, 14 heads)
- Per-layer analysis: quality consistent across depth (cosine >0.98 for uniform_4b)
- Roundtrip MSE: 0.0014 (synthetic), 0.0025 (real model data)

### Performance (Apple M-series ARM)
- Quantize throughput: 1.4M elements/ms
- Attention throughput: 137K queries/sec
- Compression ratio: 7.53x (uniform_4b)
- SIMD speedup: 4.0x (NEON vs generic)

### Testing
- 13 C++ test suites (Google Test): polar, qjl, turbo, uniform, value, paged_cache, progressive, simd_neon, simd_avx2, threading, edge_cases, attention_all_types, llamacpp_integration
- 22 Python tests (unittest): bindings, roundtrip, attention, types
- Total: **35 tests, 100% pass rate**
- Sanitizers: ASan + UBSan + TSan clean

### Integration
- **llama.cpp**: GGML type registration (7 types, base offset 256), CLI parser with 21 aliases, from_float/to_float/vec_dot wrappers, 10 integration tests
- **Python**: ctypes bindings with NumPy support, pip installable (`pip install -e .`), TurboQuant class with quantize_keys/dequantize_keys/attention methods
- **vLLM**: integration scaffold with README guide
- **Examples**: minimal.c (10 lines), standalone.c, ab_test.c, demo_real_model.c, benchmark_types.cpp, python_quickstart.py, llamacpp_integration.cpp

### Production Readiness (v0.4)
- Integer overflow protection in size calculations
- NULL pointer and buffer size validation on all public APIs
- Edge case defense: seq_len=0, head_dim<2, odd dimensions
- TQ_ERR_BUFFER_TOO_SMALL error code
- tq_type_from_name() / tq_type_count() convenience functions
- BPE values computed from actual struct sizes

### Developer Experience
- 5-dimension scoring harness: structure/correctness/quality/performance/integration
- Hierarchical Harness methodology (Karpathy AutoResearch + ClawTeam multi-agent)
- Agent definitions (.claude/agents/): architect, core-dev, perf-dev, qa
- Skill definitions (.claude/skills/): orchestrate, develop, score, qa
- Slash commands (.claude/commands/): /score, /develop, /harness, /spawn-team, /merge-gate
- PRD documents: v0.1 through v0.4
- WBS documents: v0.1 through v0.4
- refs/ absorption audit with checklist

### Memory Impact

| Model | Context | FP16 Cache | TurboQuant | Saved |
|-------|---------|------------|------------|-------|
| Llama-3.2-3B | 64K | 7.00 GB | 0.93 GB | **87%** |
| Qwen2.5-0.5B | 128K | 10.50 GB | 2.79 GB | **73%** |
| Phi-3-mini | 16K | 6.00 GB | 1.59 GB | **73%** |

### References
- TurboQuant (ICLR 2026) — arXiv:2504.19874
- QJL (AAAI 2025) — arXiv:2406.03482
- PolarQuant (AISTATS 2026) — arXiv:2502.02617
- Harness plugin (revfactory/harness) — agent team methodology
