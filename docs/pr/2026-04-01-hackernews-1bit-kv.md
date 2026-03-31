# Hacker News — 2026-04-01

## Title

Show HN: 1-bit KV cache in pure C — 10.7x compression, byte-identical output (ICLR 2026 impl)

## URL

https://github.com/quantumaikr/TurboQuant.cpp

## First comment

We implemented the TurboQuant paper (Google Research, ICLR 2026) in pure C and pushed it to the extreme: 1-bit KV cache that produces byte-identical output to 4-bit quantization.

The math: LLM attention computes inner products <q, k>. Standard quantizers minimize reconstruction error but introduce bias in inner product estimation. The paper proves that Random Hadamard Transform + sign quantization gives an unbiased estimator. At 1 bit per dimension, attention reduces to XOR + popcount.

Results on Gemma 3 4B: 30/30 test prompts produce byte-identical tokens at 1-bit vs 4-bit. KV cache shrinks from 4.4 GB to 408 MB at 32K context.

10K lines of C11, no dependencies, NEON-vectorized. Supports Gemma 3 and Qwen3.5. Reproducible benchmark included.

The counterintuitive part: 1-bit is not an approximation that "mostly works." The inner product estimator is provably unbiased, and at greedy decoding the argmax token selection is robust to the variance. We verified this empirically across math, code, knowledge, and multilingual prompts.
