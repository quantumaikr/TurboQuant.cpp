# Reddit r/LocalLLaMA — Progressive KV: FP32 quality at 3x compression

**Title:** `[Research] 128 FP32 tokens + 4-bit everything else = FP32 quality. The KV cache doesn't need uniform precision.`

**Flair:** `Research`

---

## Body

We found something surprising while building [quant.cpp](https://github.com/quantumaikr/quant.cpp), our single-header LLM inference engine:

**Keeping just the last 128 tokens' keys at FP32 while compressing everything else to 4-bit achieves FP32 quality — regardless of context length.**

Measured on Llama 3.2 3B, 3970 tokens:

| Config | PPL | vs FP32 | Memory (32K ctx) |
|---|---:|---:|---:|
| FP32 (baseline) | 19.41 | — | 7.17 GB |
| **4-bit + 128 FP32 tokens** | **19.39** | **-0.1%** | **2.33 GB** |
| 4-bit flat | 20.02 | +3.1% | 2.30 GB |

The 128-token window costs ~1.75 MB extra and recovers the entire 3.1% quality loss.

### Why does this work?

Transformer causal attention concentrates ~70% of weight on the most recent ~128 tokens — regardless of total context length. Quantization error propagates through `attention_weight × MSE`. By keeping the high-attention region at full precision, we minimize weighted error to near zero.

**The key insight: the optimal bit allocation is temporal (which tokens), not spatial (which layers).** We verified that per-layer adaptation after RHT provides only ~0.9% theoretical benefit, while per-token adaptation (the 128-token window) provides 3.2 percentage points.

### What we got wrong along the way

We initially claimed that 2-bit + 512 FP32 tokens "Pareto-dominates" flat 4-bit. This was measured at 957 tokens where the FP32 window was 53% of all tokens — misleading. At 3970 tokens (12.9% FP32), 2-bit PPL was +36.7% — much worse. We retracted the claim. The 4-bit + 128 FP32 result survived the same scrutiny.

10 self-corrections in the project's history, all found before any external report. [Full correction log in CHANGELOG](https://github.com/quantumaikr/quant.cpp/blob/main/CHANGELOG.md).

### Try it

```bash
pip install quantcpp
```

```python
from quantcpp import Model
m = Model.from_pretrained("Llama-3.2-1B", progressive=True)
print(m.ask("What is gravity?"))
```

`progressive=True` enables the 128-token FP32 window. Default `kv_compress=1` uses 4-bit for the rest. No configuration needed.

### Links

- **PyPI**: https://pypi.org/project/quantcpp/
- **GitHub**: https://github.com/quantumaikr/quant.cpp
- **Benchmark data**: `bench/results/progressive_kv_compression.md` + `attention_aware_quantization.md`
- **WASM demo**: https://quantumaikr.github.io/quant.cpp/ (189 KB, click "Try Demo")

### Discussion questions

1. Has anyone seen similar results with other engines? llama.cpp has Q4_0/Q8_0 KV but no per-token progressive approach.
2. The 128-token invariance suggests attention locality is a fundamental property of trained transformers, not architecture-specific. Would this hold for Mamba/RWKV/other architectures?
3. At what point does 4-bit itself become the bottleneck? Our 2-bit results (+36.7%) suggest 4-bit is near-optimal for the non-window region.
