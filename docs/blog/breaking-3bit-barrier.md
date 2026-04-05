# How we broke the 3-bit KV cache barrier with delta compression

*2026-04-04 -- quantumaikr/quant.cpp*

---

KV cache is the memory wall for local LLM inference. Every token you generate stores a key and value vector for every layer and every attention head. At FP16 precision, Llama 8B burns through 8 GB of KV cache at just 16K context. On an 8 GB laptop, that leaves almost nothing for the model weights themselves. You get short conversations, truncated documents, and frequent OOM crashes.

The obvious fix is quantization: store those vectors in fewer bits. We spent three months building [quant.cpp](https://github.com/quantumaikr/quant.cpp) to find out exactly how far you can push this before things break.

## The descent into fewer bits

4-bit works. We implemented a straightforward uniform min-max quantizer for KV cache keys and ran WikiText-2 perplexity on SmolLM2 1.7B. FP32 baseline: 14.63 PPL. With 4-bit keys and Q4 values: 14.57 PPL. That is -0.4%, which is within noise -- essentially free compression. For comparison, llama.cpp's Q4_0 KV cache quantization (symmetric K+V) scores +10.6% PPL on the same model. Note: llama.cpp also supports asymmetric configs like Q8_0 K + Q5_0 V (~+1% PPL, ~1.6x compression). The quality advantage at high compression (3.8x+) comes from 128-element min-max blocks, independent K/V quantization, and delta compression of adjacent keys.

3-bit is where things get ugly. Naive 3-bit uniform quantization blows up to +62% PPL. The 8 reconstruction levels simply cannot capture the post-RHT distribution with enough fidelity. We tried Lloyd-Max optimal codebooks, asymmetric ranges, per-channel scales. Nothing brought it under +40%.

2-bit is catastrophic. The attention score distribution collapses -- cosine similarity between quantized and FP32 attention drops to 0.83. The model still generates English, but it hallucinates constantly and loses track of context.

1-bit is garbage. Or so we thought.

## The bug that taught us everything

Early in development, we had a 1-bit QJL implementation that appeared to produce byte-identical output to FP32. We were ecstatic. 1-bit keys! 16x compression! We wrote it up, ran benchmarks, started planning the blog post.

Then we found the bug.

Our attention kernel had a fallback path for unquantized cache entries. During prefill, the first pass through the KV cache was writing FP32 values into the cache slots before quantization ran on them. The 1-bit "quantized" attention was actually computing against FP32 data for the entire prompt, and only using quantized values for the handful of generated tokens afterward. The FP32 prompt attention dominated the scores, masking the 1-bit noise completely.

After fixing the fallback, 1-bit key-only attention cosine dropped to 0.634 (theory predicts 2/pi = 0.637). Greedy decoding still matched on short sequences, but perplexity on longer benchmarks showed the real picture. We kept 1-bit as a supported mode because it does have legitimate uses -- the inner product estimator is provably unbiased -- but it taught us to never trust a number we had not traced end-to-end through the pipeline.

## The insight: keys are mostly redundant

We were staring at per-token key vectors, plotting them across sequence positions, when the pattern became obvious. Adjacent keys in the same layer and head are not independent. The cosine similarity between key[t] and key[t-1] averages 0.70 across layers. The difference vector -- key[t] minus key[t-1] -- has roughly 30% of the magnitude of the original.

If you have ever worked with video codecs, this is the P-frame idea. You do not store every frame as a full image. You store a keyframe (I-frame) periodically and encode the deltas in between. The deltas have lower entropy, so they compress better at the same bit budget.

We applied the same principle to KV cache keys. Store a full-precision anchor key every 64 tokens (the I-frame interval). For every token in between, quantize and store only the delta: key[t] - anchor. At decode time, reconstruct by adding the quantized delta back to the anchor.

## Delta compression results

The results on WikiText-2 with SmolLM2 1.7B, which we chose because it is small enough that anyone can reproduce on a laptop:

| Config | PPL | vs FP32 baseline (14.63) |
|--------|-----|--------------------------|
| FP32 (no compression) | 14.63 | -- |
| 4-bit K + Q4 V | 14.57 | -0.4% |
| delta + 4-bit K + Q4 V | 14.63 | +0.0% |
| delta + 3-bit K + Q4 V | 14.82 | +1.3% |
| llama.cpp Q4_0 KV | 16.18 | +10.6% |

Delta compression at 4-bit is indistinguishable from FP32. At 3-bit, the +1.3% degradation is small enough to be practical for most applications. And the memory savings are real: on an 8 GB laptop running Llama 8B with Q4 weights, KV cache compression extends usable context from roughly 16K to 61K tokens -- a 3.8x gain.

## The speed tradeoff

Delta compression is not free. Reconstructing each key requires reading the I-frame anchor and accumulating all deltas since then. On SmolLM2 1.7B (Apple M3, 4 threads): plain 4-bit runs at 25 tok/s, while delta + 3-bit drops to 7 tok/s. This is the cost of trading compute for memory. Use delta mode when context length matters more than generation speed -- long-document summarization, RAG with large retrieval windows, or offline batch processing.

## What did not work: the 2-bit wall

We spent two weeks trying to make delta compression work at 2 bits. It does not. The problem is drift. Each reconstructed key accumulates a small quantization error. When you use that reconstructed key as the anchor for the next delta, the error compounds. Per-step cosine similarity between reconstructed and original starts at 0.997 but degrades to 0.885 after 200 steps.

We tried everything: shorter I-frame intervals (every 8 tokens -- too much overhead), error feedback loops (complexity explodes), hybrid schemes mixing 2-bit deltas with 3-bit anchors. None of it crossed the threshold into usable territory. The fundamental issue is that 4 reconstruction levels cannot represent the delta distribution without systematic bias, and that bias accumulates.

3 bits appears to be the floor for delta-compressed KV cache keys that produce acceptable perplexity. We are publishing this negative result because knowing where the wall is saves everyone else the two weeks we spent hitting it.

## Try it yourself

The entire implementation is 33K lines of pure C with zero dependencies. It builds on Linux, macOS, and Windows with any C11 compiler.

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run with delta-compressed 3-bit keys
./build/quant model.gguf -p "your prompt here" -k uniform_3b -v q4 --delta

# Run with 4-bit keys (recommended default)
./build/quant model.gguf -p "your prompt here" -k uniform_4b -v q4

# Measure perplexity yourself
./build/quant model.gguf --ppl wikitext2_test.txt -k uniform_3b -v q4 --delta
```

You will need a GGUF model file. Any model from Hugging Face in GGUF format works. We tested with SmolLM2-1.7B, Llama-3.1-8B, and Qwen3.5-0.5B.

The code is at [github.com/quantumaikr/quant.cpp](https://github.com/quantumaikr/quant.cpp), Apache 2.0 licensed. If you find a bug -- especially another FP32 fallback masking real results -- please open an issue.
