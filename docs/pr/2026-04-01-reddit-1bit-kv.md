# r/LocalLLaMA — 2026-04-01

## Title

1-bit KV cache with byte-identical output to 4-bit — 10.7x compression, verified on 30 test cases (TurboQuant.cpp)

## Body

We compressed the KV cache to **1 bit per element** and got the **exact same output** as 4-bit uniform quantization. Not similar — byte-identical, token for token.

**Gemma 3 4B, greedy decode, 100 tokens:**

```
KV Type        Bits   Compression   Output
uniform_4b      4       3.8x       "Paris is the capital city of France."
turbo_kv_3b     3       4.6x       "Paris is the capital city of France."
turbo_kv_1b     1      10.7x       "Paris is the capital city of France."
                                    ↑ byte-identical
```

30 prompts tested (math, code, knowledge, Korean, long context). 30/30 identical.

**What this means at 32K context (Gemma 4B):**

```
FP16 KV:          4,352 MB
TurboQuant 1-bit:   408 MB   ← 3.9 GB saved
```

**How:** Faithful implementation of the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). Random Hadamard Transform decorrelates channels, then we just store signs. Attention becomes XOR + popcount — two CPU instructions per 128-dim key.

The key insight from the paper: MSE-optimal quantizers are **biased** for inner product estimation. TurboQuant's two-stage approach (codebook + QJL residual) corrects this bias. At 1-bit, it's purely sign-based but still **unbiased** for inner products.

**Reproduce:**
```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh
bash bench/kv_quality_bench.sh gemma3-4b.tqm
# → 30/30 byte-identical matches
```

Pure C, zero dependencies, 10K lines. Supports Gemma 3 (4B, 270M) and Qwen3.5 (0.8B).

GitHub: https://github.com/quantumaikr/TurboQuant.cpp
