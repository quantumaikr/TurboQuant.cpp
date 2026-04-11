# RLV — Read-Locate-Verify document QA

A 5-stage human-cognition-inspired document QA architecture built on top of `quant.cpp`. The challenge, motivation, architecture, and project plan are in **[`docs/phase3_rlv_challenge.md`](../../docs/phase3_rlv_challenge.md)** at the repo root — read that first if you've never seen this work.

## Quickstart

```bash
# From the repo root
python3 bench/rlv/rlv_orchestrator.py \
    --doc bench/data/wikitext2_test.txt \
    --question "Who is Robert Boulter?" \
    --model models/Llama-3.2-3B-Instruct-Q8_0.gguf
```

## Layout

```
bench/rlv/
├── README.md                    # this file
├── rlv_orchestrator.py          # main entry point
├── stages/
│   ├── __init__.py
│   ├── gist.py                  # Stage 1: chunked summarisation → outline
│   ├── locator.py               # Stage 2: outline + question → region pointer
│   ├── lookup.py                # Stage 3: region.kv + question → answer
│   ├── verifier.py              # Stage 4: gist + answer → verdict
│   └── researcher.py            # Stage 5: retry with a different region
├── prompts/                     # template prompts (gist/locator/lookup/verify)
├── eval/
│   ├── eval_acme.py             # D3: v0.12 Acme reproduction
│   └── eval_stress.py           # D5: 8000-token stress test
└── tests/
    └── smoke_test.py
```

## Cliff invariant

Every stage's prompt MUST be ≤ **1024 tokens** for Llama-3.2-3B-Q4 (the cliff measured in Phase 1B). The orchestrator enforces this in `_check_cliff_budget()`.
