# TurboQuant.cpp v1.2 PRD — From Research to Adoption

**Version**: 1.2
**Date**: 2026-04-03
**Status**: Active
**Goal**: 1000+ GitHub Stars

---

## Current State

- 50+ stars, 4 architectures verified (Llama, Gemma, Qwen, Qwen-MoE)
- PPL +0.00% at 800 tokens (1-bit K), +0.03% (1-bit K + Q4 V)
- 7.1x KV compression, 33 tests, CI green
- llama.cpp integration patch ready but not submitted
- No standard benchmarks (WikiText-2, MMLU)
- Not usable from llama.cpp/ollama/vLLM

## Problem

TurboQuant is a proven algorithm trapped in a standalone engine. The community can't use it without building from source and abandoning their existing tools. To reach 1000+ stars, the algorithm must live inside tools people already use.

## Objectives

| # | Objective | Success Metric | Priority |
|---|-----------|---------------|----------|
| O1 | llama.cpp integration works end-to-end | `--cache-type-k tq_1b` produces correct output in llama.cpp fork | P0 |
| O2 | Standard benchmark proves the claim | WikiText-2 PPL: TurboQuant 1b ≤ llama.cpp Q4 KV | P0 |
| O3 | Killer demo that goes viral | 128K context on 16GB Mac, GIF/video posted | P1 |
| O4 | One-command experience | `docker run` or `pip install` → working demo | P1 |
| O5 | Academic credibility | arXiv paper with standard benchmarks | P2 |

## Non-Goals

- Building a production serving engine (use vLLM/llama.cpp instead)
- Supporting 50+ architectures (focus on Llama family first)
- Competing on inference speed (our value is memory, not speed)
- CUDA GPU testing (requires hardware we don't have)

---

## Phase 1: llama.cpp Fork (Days 1-3)

**Goal**: TurboQuant KV working inside llama.cpp, not our engine.

1. Fork llama.cpp, apply our integration patch
2. Add GGML_TYPE_TQ_KV_1B to ggml.h
3. Register quantize/dequantize in ggml.c type_traits
4. Add to CLI: `--cache-type-k tq_1b`
5. Build and run: `./llama-cli -m llama3-8b.gguf --cache-type-k tq_1b -p "Hello"`
6. Verify: PPL matches baseline on WikiText-2

**Deliverable**: GitHub repo `quantumaikr/llama.cpp` fork with TurboQuant KV

## Phase 2: Standard Benchmarks (Days 3-5)

**Goal**: Undeniable comparison table.

1. WikiText-2 PPL (full test set, ~36K tokens):
   - llama.cpp FP16 KV (baseline)
   - llama.cpp Q4_0 KV
   - llama.cpp Q8_0 KV
   - TurboQuant 1-bit KV
   - TurboQuant 1-bit K + Q4 V

2. Memory measurement at 32K context:
   - RSS for each configuration
   - KV-only memory breakdown

3. Models: SmolLM2-1.7B (available), Qwen3.5-0.8B, Gemma-4B

**Deliverable**: `bench/results/wikitext2_comparison.md` with reproducible results

## Phase 3: Killer Demo (Days 5-7)

**Goal**: "128K context on 16GB Mac" video.

1. Prepare a 50-page public domain text (e.g., Project Gutenberg)
2. Run with `--ctx 131072 -k turbo_kv_1b -v q4`
3. Show: model reads the book, answers questions about it
4. Record: terminal screencast with memory stats overlay
5. Post to r/LocalLLM, r/MachineLearning, HN

**Deliverable**: Demo video/GIF + Reddit post

## Phase 4: Paper & Release (Days 7-14)

**Goal**: Academic credibility + easy installation.

1. Update `docs/technical_report.md` with WikiText-2 results
2. Add comparison with KIVI, Gear, KVQuant (from their papers)
3. Submit to arXiv
4. GitHub Release v1.2.0 with pre-built binaries (macOS arm64, Linux x86_64)
5. Docker image on GitHub Container Registry

**Deliverable**: arXiv paper + GitHub Release + Docker image

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|---------------|
| GitHub Stars | 200+ (path to 1000) | GitHub |
| llama.cpp fork users | 50+ clones | GitHub traffic |
| Reddit upvotes | 50+ on demo post | Reddit |
| WikiText-2 PPL delta | ≤ +0.1% vs FP16 | Benchmark script |
| Standard comparison | TQ 1b < llama.cpp Q4 memory | RSS measurement |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| llama.cpp type system incompatible | Self-contained fork, not PR |
| WikiText-2 PPL worse than expected | Test on our engine first, then port |
| 128K context OOM on 16GB | Use smaller model (0.8B) or 64K |
| arXiv rejection | Post as technical report on GitHub |
