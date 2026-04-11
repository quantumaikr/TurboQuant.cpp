# arXiv submission metadata

## Title
The Working Memory Cliff: Measuring When Quantized Edge LLMs Stop Following Instructions in Long Context

## Authors
quant.cpp maintainers (corresponding author: hi@quantumai.kr)

## Primary classification
- cs.CL (Computation and Language)

## Secondary classification
- cs.LG (Machine Learning)
- cs.AI (Artificial Intelligence)

## Keywords
needle-in-a-haystack; long-context retrieval; KV cache compression; edge-device LLM; quantization; small language models; instruction following; benchmark; reproducibility

## Abstract (arXiv format, ~280 words)

Edge-device LLM deployments increasingly rely on small (1B–3B parameter) models with aggressive weight and KV-cache quantization to fit large nominal context windows (128K+) into 16 GB consumer hardware. The "long-context inference replaces RAG" argument — load the whole document into context, skip the chunker, eliminate the silent-hallucination failure mode of vector retrieval — is correct at frontier scale and at the memory-allocation level even at the edge. We ask whether it is correct at the *retrieval* level for small quantized models that ship with `pip install`-grade inference engines.

We measure 204 needle-in-a-haystack trials on Llama-3.2-1B-Q8 and Llama-3.2-3B (both default Q4 and FP32 weight loader paths) across context lengths 256–2048 tokens, comparing FP32 KV cache baseline against 6.4× compression (`turbo_kv_4b -v q4 --k-window 128`). Both models exhibit a sharp working memory cliff at less than 1% of their nominal 128K context window: 1B Q8 transitions from 100% retrieval at 512 tokens to 0% by 1536 (graded), and 3B Q4 transitions from 100% at 1024 to 0% at 1280 *as a step function* with no degradation interval.

A six-trial FP32-weights control experiment confirms the cliff sits in the same place when on-the-fly weight requantization is disabled — going from Q4 to FP32 weights eliminates any quantization artifact but does not move the transition. The 6.4× KV cache compression is also bit-for-bit identical to FP32 baseline in 18 of 20 (model × context × method) cells. The cliff is therefore a model property, not a KV-cache or weight-quantization property.

Above the cliff, the dominant failure mode is not refusal but **synthesised hallucination** — the model fuses the planted needle into the surrounding biography, producing a coherent invented sentence. This is the same silent-hallucination failure that vector RAG produces on retrieval miss, occurring in the regime that was supposed to eliminate it.

We release the protocol, raw CSVs, per-run CLI logs, and the inference engine. The "long-context replaces RAG" framing holds for documents that fit in the model's effective working memory, which is two to three orders of magnitude smaller than the nominal context window for the configurations we measured.

## Comments
Tech report. v0.3, 2026-04-11. Reproduction scripts and raw data: https://github.com/quantumaikr/quant.cpp

## License
CC BY 4.0 (text), MIT (code)

## arXiv submission checklist (to do at submit time)

- [ ] PDF compiled and renders correctly (run `bash docs/paper/build.sh`)
- [ ] Source files (.tex + .md + figures) packaged as tar.gz
- [ ] CSV results frozen (commit hash referenced in paper)
- [ ] Reproduction commands tested in a clean clone
- [ ] All five authors / affiliations confirmed (currently single-author)
- [ ] Funding/acknowledgement section reviewed
- [ ] arXiv account ready, primary category cs.CL selected
- [ ] HF blog post draft staged for simultaneous launch
- [ ] Twitter thread queued
