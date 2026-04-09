# Progressive KV Innovation Strategy

## Core Insight (2026-04-09)

> "어텐션이 이미 알고 있는 것을 양자화도 알아야 한다."

Transformer attention naturally concentrates on recent tokens (~60-80% of weight).
Aligning KV compression precision with this attention distribution is
information-theoretically near-optimal: 128 tokens at FP32 reduces PPL
degradation from +3.8% to +0.6% at 28 KB cost.

## Measured Baseline

| Config | PPL | vs FP32 | Extra Memory |
|---|---:|---:|---:|
| FP32 | 13.56 | — | — |
| turbo_kv_4b flat | 14.08 | +3.8% | 0 |
| **progressive (k=128)** | **13.64** | **+0.6%** | **28 KB** |

## 5 Strategies (Priority Order)

### S2: Infinite Scrollback [THIS SESSION]
- Status: IN PROGRESS
- Goal: context never overflows, old tokens compressed not deleted
- Headline: "Chat for hours — no context limit, no OOM"

### S4: Compressed Persistence [NEXT]  
- Goal: save/load KV cache to disk
- Headline: "Read a document once, query it forever"

### S5: WASM Demo [NEXT]
- Goal: browser-based KV compression demo
- Headline: "Try it in your browser"

### S1: Attention-Aware Quantization [RESEARCH]
- Goal: continuous bit allocation weighted by attention
- Headline: "PPL +0.0% at 3x compression" (arXiv paper)

### S3: Layer-Adaptive Compression [INCREMENTAL]
- Goal: per-layer bit allocation
- Headline: "Every layer gets the bits it needs"

## Karpathy Loop Log

### Round 1: Progressive discovery (DONE)
- Measured k_highres=64/128/256
- Found sweet spot at 128 tokens
- PPL +3.8% → +0.6%
- Committed: bench/results/progressive_kv_compression.md

### Round 2: Python API exposure (DONE)
- Added progressive=True to Model()
- Published v0.10.0 to PyPI

### Round 3: Infinite Scrollback (DONE)
- Implemented context shift in tq_generate.c + quant.h
- Verified: SmolLM2-135M at ctx=64, 500 tokens with 9 auto-shifts
- Context never overflows — generation continues seamlessly

### Round 4: Compressed Persistence (DONE)
- quant_save_context / quant_load_context API
- QKVC file format: 64-byte header + raw compressed KV data
- Python: m.save_context("doc.kv") / m.load_context("doc.kv")
- "Read once, query forever" — verified round-trip

### Round 5: Next — S5 WASM Demo or PyPI publish v0.10.0
