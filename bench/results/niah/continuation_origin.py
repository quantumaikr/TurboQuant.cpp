#!/usr/bin/env python3
"""R3: where does the continuation start?

R1+R2 found that 84% of cliff failures are literal continuation of the
haystack ("Doctors, followed by a role in..."). This script checks
WHERE in the haystack the continuation matches, to test the hypothesis
that the model is *resuming the wikitext from a specific position*
rather than generating from a random place.

Hypotheses to distinguish:

  H1 — RESUME-AT-END: continuation starts from where the assistant turn
       was supposed to begin. The chat template anchor was too weak; the
       model defaulted to "auto-complete from end of input".

  H2 — RESUME-FROM-NEEDLE: continuation starts from somewhere near the
       inserted needle. The model partially noticed the needle but then
       resumed copying haystack from there.

  H3 — JUMP-TO-EARLY: continuation starts from somewhere early in the
       haystack (model uses recency bias to grab a fluent passage).

We measure: for each CONTINUE failure, where is the longest matching
substring in the haystack? Then bucket by haystack position quartile.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
PROXY = REPO / "bench/results/niah/redeep_proxy.json"
SUBTYPE = REPO / "bench/results/niah/redeep_subtype.json"
WIKITEXT = REPO / "bench/data/wikitext2_test.txt"

NEEDLES = [
    "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023.",
    "The launch date for Project Aurora is November 14th in San Francisco.",
    "The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water.",
]


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def reconstruct_haystack(ctx_tokens: int, needle_idx: int) -> tuple:
    """Returns (haystack_with_needle, needle_position_chars)."""
    raw = WIKITEXT.read_text(encoding="utf-8", errors="replace")
    target_chars = int(ctx_tokens * 3.6)
    hay = raw[:target_chars]
    end = hay.rfind(". ")
    if end > 0:
        hay = hay[:end + 1]
    needle = NEEDLES[needle_idx]
    desired = len(hay) // 2
    sb = hay.rfind(". ", 0, max(desired, 2))
    sb = 0 if sb < 0 else sb + 2
    return hay[:sb] + needle + " " + hay[sb:], sb


def longest_match_position(response: str, haystack: str, min_len: int = 30) -> int:
    """Return the haystack character position where the longest substring
    of `response` (length >= min_len) appears, or -1 if none found."""
    rn = normalize(response)
    hn = normalize(haystack)
    best_pos = -1
    best_len = min_len - 1
    # Walk windows of decreasing length
    for L in range(min(len(rn), 80), min_len - 1, -5):
        for i in range(0, max(1, len(rn) - L)):
            sub = rn[i:i+L]
            if not sub.strip():
                continue
            pos = hn.find(sub)
            if pos >= 0 and L > best_len:
                best_pos = pos
                best_len = L
                break  # found longest at this L
        if best_pos >= 0:
            break
    return best_pos


def main():
    with open(PROXY) as f:
        proxy = json.load(f)

    # Load all FAIL trials with their failed responses
    failures = [r for r in proxy["trials"] if r["pass"] == 0]

    # Reconstruct haystacks once per (ctx, needle)
    haystack_cache = {}
    def get_hay(ctx, needle):
        key = (ctx, needle)
        if key not in haystack_cache:
            haystack_cache[key] = reconstruct_haystack(ctx, needle)
        return haystack_cache[key]

    # For each failure, find the longest match position
    results = []
    for r in failures:
        haystack, needle_pos = get_hay(r["context"], r["needle"])
        pos = longest_match_position(r["response"], haystack)
        haylen = len(normalize(haystack))
        if pos < 0 or haylen == 0:
            position_pct = None
            relative_to_needle = None
            relative_to_end = None
        else:
            position_pct = pos / haylen
            # Approx needle position in normalized chars
            needle_pct = needle_pos / max(len(haystack), 1)
            relative_to_needle = position_pct - needle_pct
            relative_to_end = position_pct - 1.0  # negative = before end
        results.append({
            **r,
            "match_pos_pct": position_pct,
            "rel_to_needle": relative_to_needle,
            "rel_to_end":    relative_to_end,
        })

    # Distribution of match positions across all failures
    valid = [r for r in results if r["match_pos_pct"] is not None]
    print(f"Failures with detectable haystack match: {len(valid)} / {len(failures)}")
    print(f"  (those without a match — model invented enough that no 30-char substring matches)\n")

    # Bucket by quartile
    buckets = Counter()
    for r in valid:
        p = r["match_pos_pct"]
        if   p < 0.25: buckets["Q1 (0-25%)"]   += 1
        elif p < 0.50: buckets["Q2 (25-50%)"]  += 1
        elif p < 0.75: buckets["Q3 (50-75%)"]  += 1
        else:          buckets["Q4 (75-100%)"] += 1

    print("Continuation start position (longest matching substring) — quartile of haystack:")
    for q in ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]:
        n = buckets[q]
        bar = "█" * (n * 30 // max(len(valid), 1))
        print(f"  {q:<14} {n:>3}  {bar}")
    print()

    # Mean position relative to needle and end
    if valid:
        mean_pos = sum(r["match_pos_pct"] for r in valid) / len(valid)
        mean_to_end = sum(r["rel_to_end"] for r in valid) / len(valid)
        mean_to_needle = sum(r["rel_to_needle"] for r in valid) / len(valid)
        print(f"Mean match position (% of haystack):  {mean_pos:.2f}")
        print(f"Mean offset from end of haystack:      {mean_to_end:+.2f}  (negative = before end)")
        print(f"Mean offset from needle position:      {mean_to_needle:+.2f}  (positive = after needle)")
        print()

    # Hypothesis verdicts
    print("--- Hypothesis verdicts ---\n")
    if valid:
        q4_frac = buckets["Q4 (75-100%)"] / len(valid)
        q1_frac = buckets["Q1 (0-25%)"]  / len(valid)
        if q4_frac > 0.5:
            print(f"H1 (RESUME-AT-END) — SUPPORTED: {q4_frac:.0%} of continuations match")
            print( "                                 the last quartile of the haystack.")
            print( "                                 → chat template anchor failed; model")
            print( "                                   defaulted to autocomplete from prompt end.")
        elif q1_frac > 0.5:
            print(f"H3 (JUMP-TO-EARLY) — SUPPORTED: {q1_frac:.0%} of continuations match")
            print( "                                 the first quartile of the haystack.")
        else:
            # Distribution is broader — check needle proximity
            near_needle = sum(1 for r in valid if abs(r["rel_to_needle"]) < 0.1)
            if near_needle / len(valid) > 0.5:
                print(f"H2 (RESUME-FROM-NEEDLE) — SUPPORTED: {near_needle/len(valid):.0%}")
                print( "                            of continuations match within ±10% of needle position.")
            else:
                print("MIXED — no single position dominates. Continuations are spread across the haystack.")

    # Save
    out = REPO / "bench/results/niah/continuation_origin.json"
    with open(out, "w") as f:
        json.dump({
            "n_failures":      len(failures),
            "n_with_match":    len(valid),
            "buckets":         dict(buckets),
            "mean_position":   mean_pos if valid else None,
            "mean_offset_end": mean_to_end if valid else None,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
