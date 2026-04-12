#!/usr/bin/env python3
"""RLV (Read-Locate-Verify) document QA orchestrator.

Implements the 5-stage architecture from docs/phase3_rlv_challenge.md:

    Stage 1 GIST     — chunked summarisation pass → structured outline
    Stage 2 LOCATOR  — outline + question → region pointer
    Stage 3 LOOKUP   — region + question → tentative answer
    Stage 4 VERIFY   — gist + answer → {confident, unsure, contradicted}
    Stage 5 RESEARCH — retry with different region if verify fails
    Stage 6 OUTPUT   — calibrated final answer (confident or explicit uncertainty)

Cliff invariant (see docs/phase3_rlv_challenge.md §3.2): every stage's
prompt must be ≤ 1024 tokens for Llama-3.2-3B-Q4. The harness enforces
this through stages._llm.check_cliff_budget().

Usage:
    python3 bench/rlv/rlv_orchestrator.py \\
        --doc bench/data/wikitext2_test.txt \\
        --question "Who is Robert Boulter?"

For evals see bench/rlv/eval/.
"""
import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Day 2 finding: Llama-3.2-3B-Q4 in chat mode confuses ALL-CAPS acronyms
# under Q4 visual jitter. The model renders "CFO" as "ccf" and can't
# distinguish it from "CEO" → "ceoce". Result: asking "Who is the CFO?"
# returns the CEO. Asking "Who is the chief financial officer?" returns
# the right person. We pre-expand common acronyms before sending to any
# stage so the model has the full term to anchor on.
ACRONYM_EXPANSIONS = {
    r"\bCFO\b":  "chief financial officer (CFO)",
    r"\bCEO\b":  "chief executive officer (CEO)",
    r"\bCTO\b":  "chief technology officer (CTO)",
    r"\bCOO\b":  "chief operating officer (COO)",
    r"\bCIO\b":  "chief information officer (CIO)",
    r"\bCMO\b":  "chief marketing officer (CMO)",
    r"\bCDO\b":  "chief data officer (CDO)",
    r"\bHR\b":   "human resources (HR)",
    r"\bR&D\b":  "research and development (R&D)",
    r"\bIPO\b":  "initial public offering (IPO)",
}


def _expand_acronyms(text: str) -> str:
    """Expand ALL-CAPS acronyms to full term + parenthesised acronym so
    the model has both forms to match against (resilient to Q4 jitter)."""
    out = text
    for pattern, replacement in ACRONYM_EXPANSIONS.items():
        out = re.sub(pattern, replacement, out)
    return out

# Make 'stages' importable when running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stages import gist as gist_stage
from stages import locator as locator_stage
from stages import lookup as lookup_stage
from stages import verifier as verifier_stage
from stages import researcher as researcher_stage


def answer_question(
    doc_text: str,
    question: str,
    *,
    doc_id: str = "doc",
    cached_gist: gist_stage.Gist = None,
    verbose: bool = True,
) -> dict:
    """Run the full RLV pipeline. Returns a dict with the final answer
    and per-stage diagnostic info."""
    # Input validation — fail fast with clear errors
    if not doc_text or not isinstance(doc_text, str):
        return {"question": question or "", "final_answer": "[ERROR: empty or invalid document]",
                "confidence": "none", "research": {"verdict": "ERROR", "n_retries": 0, "attempts": []},
                "timings": {}, "gist_n_chunks": 0}
    if not question or not isinstance(question, str) or not question.strip():
        return {"question": "", "final_answer": "[ERROR: empty or invalid question]",
                "confidence": "none", "research": {"verdict": "ERROR", "n_retries": 0, "attempts": []},
                "timings": {}, "gist_n_chunks": 0}

    t_start = time.time()
    timings = {}

    # Pre-process the question: expand acronyms (CFO -> "chief financial
    # officer (CFO)") so Q4 visual jitter doesn't confuse the model.
    original_question = question
    question = _expand_acronyms(question)
    if verbose and question != original_question:
        print(f"[preprocess] expanded acronyms: {original_question!r} -> {question!r}")

    # Stage 1: GIST (or use cached one)
    t0 = time.time()
    if cached_gist is not None:
        gist = cached_gist
        if verbose:
            print(f"[stage 1] using cached gist ({len(gist.chunks)} chunks)")
    else:
        if verbose:
            print(f"[stage 1] building gist for doc_id={doc_id}, len={len(doc_text)} chars")
        gist = gist_stage.build_gist(doc_text, doc_id=doc_id, verbose=verbose)
    timings["stage1_gist"] = time.time() - t0

    # Stage 2: LOCATOR
    t0 = time.time()
    if verbose:
        print(f"[stage 2] locating question: {question!r}")
    region = locator_stage.locate(question, gist, verbose=verbose)
    timings["stage2_locator"] = time.time() - t0
    if verbose:
        print(f"[stage 2] -> chunk {region.chunk_id} (confidence={region.confidence})")

    # Stage 3: LOOKUP
    t0 = time.time()
    if verbose:
        print(f"[stage 3] reading chunk {region.chunk_id}")
    look = lookup_stage.lookup(question, region, doc_text, verbose=verbose)
    timings["stage3_lookup"] = time.time() - t0
    if verbose:
        print(f"[stage 3] -> answer: {look.answer[:80]!r}")

    # Stage 4: VERIFY (citation-grounded against the lookup region)
    t0 = time.time()
    if verbose:
        print(f"[stage 4] verifying answer against region (citation-grounded)")
    ver = verifier_stage.verify(
        question, look.answer, gist,
        region_text=look.region_text,
        chunk_id=look.chunk_id,
        verbose=verbose,
    )
    timings["stage4_verifier"] = time.time() - t0
    if verbose:
        print(f"[stage 4] -> verdict: {ver.verdict} ({ver.reason})")

    # Stage 5: RESEARCH (only if verify failed)
    t0 = time.time()
    research = researcher_stage.research(
        question, look, ver, gist, doc_text, verbose=verbose,
    )
    timings["stage5_research"] = time.time() - t0

    # Stage 6: OUTPUT — format the final answer based on the verdict
    if research.final_verdict == "CONFIDENT":
        final_text = research.final_answer
        confidence = "high"
    elif research.final_verdict == "EXHAUSTED":
        final_text = (
            f"I'm not fully confident in any answer to your question. The closest "
            f"information I found is: {research.final_answer}"
        )
        confidence = "low"
    else:
        final_text = research.final_answer
        confidence = "medium"

    timings["total"] = time.time() - t_start

    return {
        "question": question,
        "final_answer": final_text,
        "confidence": confidence,
        "research": {
            "verdict": research.final_verdict,
            "n_retries": research.n_retries,
            "attempts": research.attempts,
        },
        "timings": timings,
        "gist_n_chunks": len(gist.chunks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, type=Path,
                        help="Path to the document text file")
    parser.add_argument("--question", required=True, type=str,
                        help="The question to answer")
    parser.add_argument("--doc-id", default=None, type=str)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-stage diagnostics")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human text")
    args = parser.parse_args()

    doc_text = args.doc.read_text(encoding="utf-8", errors="replace")
    doc_id = args.doc_id or args.doc.stem

    result = answer_question(
        doc_text, args.question,
        doc_id=doc_id, verbose=not args.quiet,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("\n" + "=" * 70)
        print(f"QUESTION:    {result['question']}")
        print(f"ANSWER:      {result['final_answer']}")
        print(f"CONFIDENCE:  {result['confidence']}")
        print(f"VERDICT:     {result['research']['verdict']}")
        print(f"RETRIES:     {result['research']['n_retries']}")
        print(f"GIST CHUNKS: {result['gist_n_chunks']}")
        print(f"TOTAL TIME:  {result['timings']['total']:.1f}s")
        print("  " + " | ".join(
            f"{k}={v:.1f}s" for k, v in result["timings"].items() if k != "total"
        ))
        print("=" * 70)


if __name__ == "__main__":
    main()
