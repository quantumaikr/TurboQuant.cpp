# Phase 3: Read-Locate-Verify (RLV) — The Challenge

> **Status**: Day 1 of 7 — Project documentation + Karpathy loop tracker.
> If you are a new Claude Code session reading this for the first time, this document is the canonical source of truth. Read this first, then check the `Status & Karpathy log` section at the bottom for current state.

---

## 1. The challenge in one sentence

**Build a multi-stage document QA architecture that breaks through both vector RAG's silent hallucination and long-context inference's working memory cliff, by mapping the human cognitive retrieval pattern (gist → locate → lookup → verify → re-search → calibrate) onto orchestration above the existing quant.cpp inference engine.**

## 2. Why this challenge, why now

### 2.1 The user's framing (2026-04-12)

The user explicitly pivoted from academic publishing back to practical impact:

> "이제 논문에 너무 집중하지 말고, 실질적인 성능 개선 혹은 혁신적 방법론을 찾아서 실질적인 효용성에 집중하도록 하겠습니다. 우리가 원하는건 심플합니다. 기존 RAG 한계를 돌파하는 것입니다."
>
> "사람들도 모든것을 기억할수 없습니다. 대략, 어떻다는것을 알고, 그 정보가 어디쯤에 있다는 것을 알고, 해당 정보를 찾아서, 내가 대략 알고있는것과 맞는지 애매한지를 판단하고, 다시 찾거나 합니다."

The user pointed out that humans don't remember everything either — they have a sophisticated retrieval pattern that current LLM systems only partially replicate.

### 2.2 Why vector RAG and long-context both fail

| | Vector RAG (FAISS+chunks) | Long-Context Inference |
|---|---|---|
| **Failure mode** | Silent hallucination on retrieval miss | Working memory cliff (Phase 1B finding) |
| **Root cause** | No verification step — trusts retrieved chunks | Chat-template anchor collapses at <1% of nominal context |
| **What's missing** | Phase 1 (gist) and Phase 4 (verification) of human pattern | Phase 2 (locator) and Phase 5 (re-search) |

Both systems implement *fragments* of the human pattern. **Neither implements the complete iterative loop with verification.** That is the gap RLV fills.

### 2.3 What three quant.cpp Phase results converge here

| Phase | Finding | Implication for RLV |
|---|---|---|
| **1B** | Cliff is a model property, sits at ~1% of nominal context, KV compression is bit-for-bit identical to baseline | The cliff is *external* to KV — we can avoid it by sizing each stage's prompt below it |
| **2B** | Cliff failure is "primacy-biased document continuation overflow" — *opposite* of RAG hallucination; 84% of failures are literal haystack continuation, not parametric invention | Mechanically distinct from RAG — needs different mitigation |
| **2C** | Two prompt-level anchor-strengthening interventions (PQRI, convchunk) failed to move the cliff and performed worse than baseline | Fighting the cliff at the prompt level doesn't work; *avoidance* is the productive direction |

The Phase 2C negative result is the *justification* for RLV: if even direct prompt-level reminders can't override the document-continuation prior, then the only remaining productive direction is to never let any single LLM call exceed the model's effective working memory.

## 3. The RLV architecture

### 3.1 Stage definitions

| Stage | Human cognitive phase | LLM implementation | Cliff-safe? |
|---|---|---|---|
| **1. GIST** | Build mental index of structure + topics | Chunked summarisation → structured outline (~500-2000 tokens), saved as `.gist` artifact | ✅ each chunk sized below cliff |
| **2. LOCATOR** | Recall rough position | LLM call: outline + question → region pointer `(start_token, end_token)` + confidence | ✅ outline is small |
| **3. LOOKUP** | Read targeted region in detail | Load only the targeted region's KV cache (mmap from `.kv` file), generate answer | ✅ region sized below cliff |
| **4. VERIFY** | Cross-check answer against gist | LLM call: gist + answer → `{confident, unsure, contradicted}` | ✅ both inputs small |
| **5. RE-SEARCH** | Iterate if verification fails | If verdict ≠ confident, locator retries with a different region. Capped at N=3 | ✅ same as stages 2-4 |
| **6. HONEST OUTPUT** | Calibrated "I don't know" | If all retries fail, return explicit uncertainty + relevant gist content | N/A |

### 3.2 The two non-negotiable invariants

1. **Cliff invariant**: Every single LLM call's prompt MUST be smaller than the measured effective working memory of the model being used. For Llama-3.2-3B-Q4 on the default loader path, that means **prompts must be ≤ 1024 tokens**.

2. **Verification invariant**: No answer is returned to the user unless it has passed Stage 4 verification OR has been explicitly flagged as uncertain.

These two invariants are what give RLV its hallucination resistance.

### 3.3 What quant.cpp gives us specifically

| Capability | Used in | Why it matters |
|---|---|---|
| `save_context` / `load_context` (`.kv` files) | Stage 1 (one-time gist build) and Stage 3 (region mmap) | Each region's KV cache precomputed once. Stage 3 inference is generation-bound, not prefill-bound. No other inference engine has this. |
| Phase 1B cliff measurement (1024 tokens for 3B Q4) | All stages — used as the cliff-safe budget | We measured this directly. The architecture parameter is ours. |
| 6.4× KV compression (`turbo_kv_4b -v q4 --k-window 128`) | Stage 1 and Stage 3 | Region fits in tight memory |
| Single-header `quant.h` | Subprocess orchestration from Python | No model-internal hooks needed for the prototype |

No other inference engine has all four pieces in one place.

## 4. The one-week prototype plan

### 4.1 Day-by-day

| Day | Goal | Success criterion (Karpathy gate) |
|---|---|---|
| **1** | Project documentation + memory (this doc) + harness skeleton | This doc exists, memory entry exists, `bench/rlv/` directory created with skeleton files |
| **2** | Stage 1 (gist) + Stage 2 (locator) + Stage 3 (lookup) end-to-end on one trivial example | Pipeline returns *some* answer to one question on the v0.12 Acme document |
| **3 (critical)** | Reproduce v0.12 Acme 7-question benchmark with full RLV (all 5+1 stages) | **RLV scores 7/7 matching pure long-context. If <7/7 → diagnose stage failure and fix before D4.** |
| **4** | 8000-token wikitext stress test setup: pick article, write 10 questions (5 single-hop + 5 multi-hop), build vector RAG baseline | All three systems (vector RAG, pure long-context, RLV) run on the stress test; raw outputs captured |
| **5 (critical)** | Stress test results + analysis | **RLV beats both vector RAG and pure long-context on the stress test.** Specifically: RLV ≥ vector RAG single-hop, RLV >> vector RAG multi-hop, RLV >> pure long-context overall, RLV hallucination rate < both. |
| **6** | Tech report §4.8 RLV results + master_table.md update + commit | §4.8 added to `docs/paper/working-memory-cliff.md`; commit pushed |
| **7** | Community release: HF blog draft, Reddit post draft, optional arXiv supplement | Drafts in `docs/pr/`; ready for user to publish |

### 4.2 What "success" looks like

Minimum viable success:
- **D3**: RLV matches v0.12 (7/7 on Acme).
- **D5**: RLV ≥ vector RAG and ≥ pure long-context on stress test, with hallucination rate < both.

Full success:
- **D5**: RLV ≥ 90% on stress test, vector RAG ≤ 60%, pure long-context = 0%, hallucination rate = 0% for RLV (all uncertainties explicit).

If we hit even minimum viable success the project is publishable as a practical system. Full success would be a clean industrial story.

### 4.3 What "failure" looks like, and what to do

If **D3** fails (RLV < 7/7 on Acme): diagnose which stage broke. Most likely candidates:
- **Stage 1 gist quality**: gist is too lossy. Fix: more structured template, smaller chunks.
- **Stage 2 locator accuracy**: locator points to wrong region. Fix: include keyword anchors, use multi-candidate output.
- **Stage 4 verifier**: false-confirms wrong answers OR false-flags correct answers. Fix: stricter prompt, fail-closed bias.

If **D5** fails (RLV doesn't beat both baselines): the architecture is unsound. Pivot to:
- A simpler 3-stage variant (gist → lookup → verify, no locator or re-search)
- OR document the negative result and move to attention-level interventions (multi-week project)

## 5. Files and directory layout

```
docs/
├── phase3_rlv_challenge.md              # This file. Canonical source of truth.
└── paper/
    └── working-memory-cliff.md          # §8 motivates RLV; §4.8 will land RLV results

bench/
├── rlv/                                  # NEW — created Day 1
│   ├── README.md                         # short summary, points back to this doc
│   ├── rlv_orchestrator.py               # Main entry point: question + doc → answer
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── gist.py                       # Stage 1: chunked summarisation → outline
│   │   ├── locator.py                    # Stage 2: outline + Q → region pointer
│   │   ├── lookup.py                     # Stage 3: region.kv + Q → answer
│   │   ├── verifier.py                   # Stage 4: gist + answer → verdict
│   │   └── researcher.py                 # Stage 5: retry with different region
│   ├── prompts/                          # template prompts for each stage
│   ├── eval/                             # eval scripts and result CSVs
│   │   ├── eval_acme.py                  # D3: v0.12 Acme reproduction
│   │   ├── eval_stress.py                # D5: 8000-token wikitext stress test
│   │   └── results_*.csv                 # raw measurements
│   └── tests/                            # smoke tests for each stage
└── results/niah/                         # existing — Phase 1B/2B/2C measurements
```

## 6. Reading order if you are a new Claude Code session

1. **This file (`docs/phase3_rlv_challenge.md`)** — read end to end first.
2. **The status section at the bottom** — see what's been done and what's next.
3. **Auto-memory entry** at `~/.claude/projects/-Users-bruce-Dev-projects-quant-cpp/memory/project_phase3_rlv.md` — short version of this file.
4. **`docs/paper/working-memory-cliff.md` §8** — the academic context that motivates RLV.
5. **`bench/results/niah/master_table.md`** — the Phase 1B/2B/2C measurements that the cliff invariant depends on.
6. **Latest commit on `main`** — `git log --oneline -10` to see what shipped.

## 7. The canonical "what model and what budget" reference

- **Model**: `models/Llama-3.2-3B-Instruct-Q8_0.gguf` (default loader path which auto-converts to Q4)
- **Cliff budget**: **1024 tokens** per LLM call's prompt. Verified across Phase 1B grids.
- **Inference binary**: `./build_metal/quant`
- **Reproduction command pattern**:
  ```bash
  ./build_metal/quant models/Llama-3.2-3B-Instruct-Q8_0.gguf \
      -p "<prompt>" -n 64 -T 0.0 -j 8 \
      --chat --ctx 2048 -k turbo_kv_4b -v q4 --k-window 128
  ```
- **Other safe model**: `models/Llama-3.2-1B-Instruct-Q8_0.gguf` (cliff = 512 tokens, smaller budget but faster)

---

## 8. Status & Karpathy log

The Karpathy loop log lives in this section. Every round of work updates this with: what was done, what was measured, what was decided.

### Day 1 — 2026-04-12

**Started**: 2026-04-12

**R1: Project documentation + memory persistence** ✅
- Created this canonical project doc
- Created memory entry `~/.claude/projects/.../memory/project_phase3_rlv.md` so future sessions can find this work
- Updated `MEMORY.md` index with the new entry as the top item
- Goal: any future Claude Code session that loads this project can be productive within 5 minutes by reading the memory entry → this doc → Karpathy log section

**R2: Harness skeleton** — pending
- Create `bench/rlv/` directory tree
- Stub each stage file with the function signature and a placeholder
- Add a `README.md` that points back to this doc
- Goal: `python3 bench/rlv/rlv_orchestrator.py --doc ... --question ...` runs end-to-end without crashing (returns a placeholder answer)

**R2: Harness skeleton** ✅
- Created `bench/rlv/` with stages/, prompts/, eval/, tests/ subdirs
- Implemented `_llm.py`, `gist.py`, `locator.py`, `lookup.py`, `verifier.py`, `researcher.py`, `rlv_orchestrator.py`
- Implemented `tests/smoke_test.py` (the D1 gate)

**R3: First end-to-end pipeline** ✅ — integration only, accuracy is D2
- Initial run: model reload per subprocess call → 5 minutes per question. Pivoted to using `quant-server` HTTP endpoint, model loaded once.
- Built `quant-server` with `cmake --build build_metal --target quant-server`.
- Refactored `_llm.py` to use `start_server()`/`stop_server()` + `/v1/chat/completions` HTTP API. Now ~10 sec per call instead of 50.
- Multiple parser fixes:
  - Subprocess stdout/stderr were being concatenated wrong; switched to `stderr=STDOUT` to get bash `2>&1`-like merging.
  - 3B Q4 in chat mode emits `## Step 1: ...` reasoning chains instead of structured format. Diagnosed and added a short system prompt: `"Answer in one short sentence. No reasoning steps."`
  - Redesigned all 4 stage prompts from "structured format" (TOPICS:/CHUNK:/VERDICT:) to direct natural-language questions, with tolerant parsers that extract answers from verbose responses.
- Diagnosed primacy bias on the smoke test doc: even at 270 chars (well below cliff), the model picks the first-mentioned entity (Maria Santos / CEO) when asked about CFO. **This is exactly the Phase 2B finding** — and it's exactly what RLV is supposed to fix by isolating each chunk.
- Bumped chunk_chars from 2400 to 500 so even small docs split into multiple chunks for the locator to choose from.

**D1 gate verdict**: integration ✅ (pipeline runs end-to-end without crashing), accuracy ❌ (smoke test still picks wrong entity).

**Open issue for Day 2**: gist summaries are too vague to discriminate ("This section is about Acme Robotics, a company that..." for every chunk). The locator picks the wrong chunk, so the lookup reads the wrong region.

**D2 plan**: switch the locator's index from "model-written summary" to "first ~100 chars of each chunk's actual text". This gives the locator real semantic signal and avoids depending on the model's summarization quality. Direct extraction beats LLM summarization for indexing.

---

### Day 2 — pending
