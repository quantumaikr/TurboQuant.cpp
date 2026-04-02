# TurboQuant.cpp — Work Breakdown Structure v1.2

**Version**: 1.2
**Date**: 2026-04-03
**Focus**: llama.cpp Integration, Standard Benchmarks, Killer Demo

---

## Phase 1: llama.cpp Fork Integration (Days 1-3)

### 1.1 Fork Setup (~2h)
- [ ] Fork ggerganov/llama.cpp to quantumaikr/llama.cpp
- [ ] Clone fork locally alongside TurboQuant.cpp
- [ ] Build baseline llama.cpp: `cmake -B build && cmake --build build`
- [ ] Verify baseline works: `./build/bin/llama-cli -m model.gguf -p "Hello"`

### 1.2 Add GGML Type (~4h)
- [ ] Copy `integrations/llamacpp/patch/ggml-turbo-quant.h` to `ggml/include/`
- [ ] Copy `integrations/llamacpp/patch/ggml-turbo-quant.c` to `ggml/src/`
- [ ] Add `GGML_TYPE_TQ_KV_1B = 41` to `ggml/include/ggml.h` enum
- [ ] Increment `GGML_TYPE_COUNT` to 42
- [ ] Add type_traits entry in `ggml/src/ggml.c`:
  ```c
  [GGML_TYPE_TQ_KV_1B] = {
      .type_name = "tq_kv_1b",
      .blck_size = 128,
      .type_size = 24,
      .is_quantized = true,
      .to_float = dequantize_row_tq_kv_1b,
      .from_float_ref = quantize_row_tq_kv_1b_ref,
  }
  ```
- [ ] Add source to CMakeLists.txt: `ggml/src/ggml-turbo-quant.c`
- [ ] Build: verify no compile errors

### 1.3 Enable CLI (~2h)
- [ ] Add `GGML_TYPE_TQ_KV_1B` to `kv_cache_types` in `common/arg.cpp`
- [ ] Build and test: `./build/bin/llama-cli -m model.gguf --cache-type-k tq_kv_1b -p "Hello"`
- [ ] Verify output is coherent (not garbage)

### 1.4 PPL Verification (~2h)
- [ ] Run PPL: `./build/bin/llama-perplexity -m model.gguf --cache-type-k tq_kv_1b`
- [ ] Compare vs `--cache-type-k f16` (baseline)
- [ ] Compare vs `--cache-type-k q4_0` (llama.cpp native Q4)
- [ ] Record results in `bench/results/llamacpp_ppl.md`

### 1.5 Fix Issues (~4h)
- [ ] If PPL is wrong: debug quantize/dequantize path
- [ ] If crashes: check block size alignment, type registration
- [ ] If slow: profile dequantize overhead
- [ ] Iterate until PPL delta < 0.1%

---

## Phase 2: Standard Benchmarks (Days 3-5)

### 2.1 WikiText-2 Setup (~2h)
- [ ] Download WikiText-2 test set
- [ ] Convert to plain text format for llama-perplexity
- [ ] Verify baseline PPL matches published numbers (±10%)

### 2.2 Comprehensive PPL Table (~4h)
- [ ] Measure on SmolLM2-1.7B (or available Llama-family model):

| Config | KV Memory/token | WikiText-2 PPL |
|--------|----------------|----------------|
| FP16 KV (baseline) | 256 bytes | ? |
| llama.cpp Q8_0 KV | 128 bytes | ? |
| llama.cpp Q4_0 KV | 64 bytes | ? |
| **TurboQuant 1-bit K** | **24 bytes** | **?** |

- [ ] Measure memory: `vmstat` or `/proc/self/status` during inference
- [ ] Create comparison chart (ASCII or markdown)

### 2.3 Memory Crossover Chart (~2h)
- [ ] Measure RSS at context lengths: 1K, 4K, 8K, 16K, 32K, 64K
- [ ] For each KV type: FP16, Q4, TQ_1b
- [ ] Find the crossover point where FP16 OOMs but TQ_1b survives
- [ ] Create chart for README

### 2.4 Publish Results (~1h)
- [ ] Write `bench/results/wikitext2_comparison.md`
- [ ] Update README with benchmark table
- [ ] Commit + push

---

## Phase 3: Killer Demo (Days 5-7)

### 3.1 Long Context Setup (~2h)
- [ ] Prepare 50-page text (Project Gutenberg, public domain)
- [ ] Tokenize and verify: ~30K-50K tokens
- [ ] Test: model loads + generates with `--ctx 65536`

### 3.2 Demo Script (~2h)
- [ ] Create `scripts/demo_long_context.sh`:
  ```bash
  # Shows: load book → ask question → get answer → show memory
  ./build/tq_run model.gguf \
    --ctx 65536 -k turbo_kv_1b -v q4 \
    -p "$(cat book.txt) \n\nSummarize the key themes:" \
    -n 200 -M
  ```
- [ ] Test on SmolLM2 1.7B (fits in 16GB with 64K context)

### 3.3 Record Demo (~2h)
- [ ] Install asciinema or screen recorder
- [ ] Record: build → load model → long context generation → memory stats
- [ ] Convert to GIF (< 5MB for Reddit)
- [ ] Upload to GitHub repo

### 3.4 Community Post (~2h)
- [ ] Write Reddit post: "128K context on 16GB Mac — 7x KV compression with zero PPL loss"
- [ ] Include: benchmark table, GIF, GitHub link
- [ ] Post to r/LocalLLM, r/MachineLearning
- [ ] Prepare answers for expected questions

---

## Phase 4: Paper & Release (Days 7-14)

### 4.1 Paper Update (~8h)
- [ ] Add WikiText-2 results to `docs/technical_report.md`
- [ ] Add llama.cpp comparison section
- [ ] Add comparison vs KIVI, Gear (from their published numbers)
- [ ] Format for arXiv submission
- [ ] Internal review

### 4.2 GitHub Release (~2h)
- [ ] Tag: `git tag -a v1.2.0 -m "Release v1.2.0"`
- [ ] Build binaries: macOS arm64, Linux x86_64
- [ ] Create GitHub Release with:
  - Pre-built binaries
  - Benchmark results
  - llama.cpp fork link
  - Quick start guide

### 4.3 Docker Image (~1h)
- [ ] Build: `docker build -t ghcr.io/quantumaikr/turboquant .`
- [ ] Push to GHCR
- [ ] Test: `docker run ghcr.io/quantumaikr/turboquant model.gguf -p "Hello" -k turbo_kv_1b`

### 4.4 Announcement (~2h)
- [ ] Update README with release badge
- [ ] Post to HN: "Show HN: 1-bit KV Cache — 7x memory reduction, zero PPL loss"
- [ ] Tweet/post with benchmark chart
- [ ] Submit paper to arXiv

---

## Verification Checkpoints

| Checkpoint | Criteria | When |
|-----------|---------|------|
| **V1** | llama.cpp fork builds with TQ type | Day 1 |
| **V2** | `--cache-type-k tq_kv_1b` produces coherent output | Day 2 |
| **V3** | WikiText-2 PPL delta < 0.1% vs FP16 | Day 4 |
| **V4** | Memory table shows TQ < Q4 < Q8 < FP16 | Day 4 |
| **V5** | 64K+ context demo works on 16GB | Day 6 |
| **V6** | GitHub Release published | Day 8 |
| **V7** | arXiv paper submitted | Day 14 |

---

## Resource Requirements

- llama.cpp fork: ~1 day setup
- WikiText-2 dataset: free download
- Model: SmolLM2-1.7B (already downloaded), Qwen 0.8B (available)
- Hardware: M3 Mac Air 16GB (available)
- No CUDA GPU needed (CPU benchmarks sufficient for PPL comparison)
