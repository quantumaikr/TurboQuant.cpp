# Reddit Post — quant.cpp KV quality comparison (2026-04-06)

Target: r/LocalLLaMA, r/LocalLLM

**Title:** Same 4 bits. Very different quality. (quant.cpp vs llama.cpp KV compression)

**Post type:** Image + text

**Image:** How It Compares chart (PPL bar chart + engine comparison table)

---

**Body (below image):**

Both use 4-bit KV quantization. Same bit budget, but very different quality outcomes.

**WikiText-2 PPL (SmolLM2 1.7B):**

- llama.cpp Q4_0 K+V: PPL **+10.6%**
- quant.cpp 4-bit K + Q4 V: PPL **+0.0%**
- quant.cpp 3-bit delta K + Q4 V: PPL **+1.3%**

**Why the difference?** Both are per-block methods, but with different design choices:

- **Block size & range encoding**: llama.cpp Q4_0 uses 32-element blocks with a single zero-point scale. quant.cpp uses 128-element blocks with min-max range encoding, which better captures the distribution of key vectors specifically.
- **Independent K/V treatment**: quant.cpp applies different quantization methods to keys vs values, optimized for each tensor's statistical properties.
- **Delta compression** (unique to quant.cpp): stores `key[t] - key[t-1]` instead of absolute keys — like video P-frames. Adjacent keys differ by ~30% of their range, so 3-bit deltas work where absolute 3-bit gives +62% PPL.

**Fair note**: llama.cpp also supports separate K/V quant types. Q8_0 K + Q5_0 V is a solid config with much less degradation than Q4_0 on both — but at ~1.6x compression. quant.cpp targets the 4-7x range (extending 50K context to 350K) where the quality gap matters most.

On a 16GB Mac with Llama 3.2 3B: llama.cpp with FP16 KV maxes out at ~50K tokens. quant.cpp compresses KV 6.9x → **~350K tokens** with zero quality loss.

Not trying to replace llama.cpp — it's faster. Use llama.cpp for speed, vLLM for throughput, quant.cpp when context length is your bottleneck.

72K LOC, pure C, zero dependencies. Also ships as a single-header `quant.h` (15K LOC).

Source: [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp)

---

## 수정 이력

**v2 (audioen 피드백 반영):**
- "llama.cpp applies the same scheme to both K and V" 삭제 — 부정확 (llama.cpp도 별도 설정 가능)
- "per-block" 공통점 인정, block size/range encoding 차이로 정정
- llama.cpp Q8_0 K + Q5_0 V 옵션을 공정하게 언급
- "breaks the model" → "very different quality outcomes" 톤 완화
- 전체적으로 팩트 기반으로 재작성, 과장 표현 제거

**v1 문제점 (삭제된 표현):**
- ~~"One breaks the model, the other doesn't"~~ → 과장
- ~~"llama.cpp applies the same Q4_0 scheme to both keys and values"~~ → 부정확
- ~~"Outliers stay local instead of corrupting the whole tensor"~~ → Q4_0도 per-block이므로 misleading
