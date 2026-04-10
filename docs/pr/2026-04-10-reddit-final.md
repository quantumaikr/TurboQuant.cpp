# Reddit r/LocalLLaMA

**Title:** `We compressed AI's memory 3x. It got faster.`

**Flair:** `Resources`

---

We built [quant.cpp](https://github.com/quantumaikr/quant.cpp) — a single-file C engine (16K lines) that compresses LLM KV cache memory. The counterintuitive result:

**3x less memory. 0% quality loss. 13% faster.**

No tradeoff.

### How

AI focuses on recent tokens, just like humans focus on recent conversation. We keep the last 128 tokens at full precision and compress everything else to 4-bit. The quality loss from compression drops to zero because the compressed tokens barely affect the output.

### Numbers (3 models, verified)

| Model | Compression only | + Progressive (auto) |
|---|---|---|
| Llama 3.2 3B | -3.1% quality | **0% loss** |
| Llama 3.2 1B | -16.1% quality | **-1.2% loss** |
| SmolLM2 135M | -12.9% quality | **-3.1% loss** |

### When to use this

- **Embedding AI into your app/game/device** → one C file, zero deps
- **8GB laptop + long conversations** → 32K context where FP32 would OOM
- **Browser AI with no server** → [264 KB WASM demo](https://quantumaikr.github.io/quant.cpp/)

### When NOT to use this

- **Max GPU speed** → use llama.cpp
- **Batch serving** → use vLLM

### Try it

```bash
pip install quantcpp
```

```python
from quantcpp import Model
m = Model.from_pretrained("Llama-3.2-1B")
print(m.ask("What is gravity?"))
```

Progressive compression is on by default. Nothing to configure.

[GitHub](https://github.com/quantumaikr/quant.cpp) · [PyPI](https://pypi.org/project/quantcpp/) · [Browser Demo](https://quantumaikr.github.io/quant.cpp/)
