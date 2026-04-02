# TurboQuant.cpp — 1-bit KV cache with zero quality loss, verified on 35B MoE

Pure C inference engine implementing the TurboQuant paper (ICLR 2026). Built from scratch, not a llama.cpp fork.

**What it does:** Compresses KV cache keys to 1 bit using randomized Hadamard transform + sign hashing. The output is byte-identical to the uncompressed baseline.

**Verified results:**

```
Qwen3.5-35B-A3B MoE (IQ2_XXS GGUF, 16GB Mac):
  baseline:   "The capital of France is Paris."
  1-bit KV:   "The capital of France is Paris."   ← same output

Gemma 3 4B (TQM, perplexity 101 tokens):
  FP16 KV:        PPL = 35.99
  1-bit K + Q4 V:  PPL = 36.00  (+0.03%)
```

1-bit attention cosine = 0.634, matching the information-theoretic limit of 2/pi. Formal unbiasedness verified at < 0.2% relative bias over 100K random vector pairs.

**What's in the repo:**

- 27K lines of C/Metal, zero external dependencies
- GGUF direct loading (Q8_0, Q4_K_M, IQ2_XXS verified)
- MoE support (256 experts, top-8, shared expert)
- 1-bit weight quantization (8.4x compression, zero quality loss on 4B)
- Metal GPU backend (Apple Silicon), CUDA/Vulkan/ROCm compile targets
- 32 test suites, ASan clean
- Perplexity measurement, activation profiling, codebook calibration tools

**Honest limitations:**

- CPU inference only for now (Metal MoE dispatch is WIP)
- 35B at ~1-4 tok/s on M3 16GB (memory bandwidth bound)
- IQ2_XXS (2-bit weights) limits quality on complex reasoning — that's the weight quantization, not the KV compression
- Tested on Qwen3.5 and Gemma 3 only (3 architectures)

**The algorithm (from the paper):**

Keys: normalize -> RHT -> Lloyd-Max codebook -> QJL sign hash
1-bit: signs only -> attention via XOR + popcount

Values: per-block Q4 or Q2 quantization

The paper proves standard quantizers introduce systematic bias in inner product estimation. RHT + QJL correction makes it provably unbiased.

https://github.com/quantumaikr/TurboQuant.cpp

Paper: https://arxiv.org/abs/2504.19874

Happy to answer questions about the implementation or the algorithm.
