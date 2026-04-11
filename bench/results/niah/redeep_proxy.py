#!/usr/bin/env python3
"""ReDeEP-Cliff hypothesis test — surface-level proxy.

Tests whether cliff hallucinations have the same signature as RAG
hallucinations described by ReDeEP (Sun et al., ICLR 2025): low External
Context Score + high Parametric Knowledge Score.

We don't have direct access to attention heads and FFN residuals from
quant.cpp's forward pass without instrumenting the C code, so we use
**surface-level proxies** computed from response text alone:

  copy_score(response, haystack)
    = fraction of response 4-grams that appear in the haystack
    = proxy for "model is copying from the loaded context"
    = proxy for ReDeEP's External Context Score

  novel_score(response, haystack)
    = fraction of response 4-grams that DO NOT appear in the haystack
    = proxy for "model is generating from parametric knowledge"
    = proxy for ReDeEP's Parametric Knowledge Score

Note: copy_score + novel_score = 1.0 by construction. We report both for
clarity.

Hypothesis: at the cliff (model fails to retrieve), copy_score should
DROP and novel_score should RISE compared to the pre-cliff regime where
the model retrieves correctly. This would mirror what ReDeEP observed
for RAG hallucinations: External Context Score low, Parametric
Knowledge Score high.

Usage:
    python3 bench/results/niah/redeep_proxy.py
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
CSVS = [
    REPO / "bench/results/niah/results_20260411T024534.csv",  # 3B Q4 R1 (512+1024)
    REPO / "bench/results/niah/results_20260411T043236.csv",  # 1B Q8 sweep
    REPO / "bench/results/niah/results_20260411T052319.csv",  # 3B Q4 ceiling probe
]
WIKITEXT = REPO / "bench/data/wikitext2_test.txt"

# Same three needles as bench/niah_test.sh
NEEDLES = [
    "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023.",
    "The launch date for Project Aurora is November 14th in San Francisco.",
    "The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water.",
]


def normalize(s: str) -> str:
    """Lowercase + strip non-alphanum."""
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def ngrams(text: str, n: int = 4) -> set:
    """Set of n-grams (whitespace tokens)."""
    toks = normalize(text).split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def reconstruct_haystack(ctx_tokens: int, needle_idx: int) -> str:
    """Reconstruct the exact haystack the trial saw — same logic as
    bench/niah_test.sh's build_prompt python helper."""
    raw = WIKITEXT.read_text(encoding="utf-8", errors="replace")
    target_chars = int(ctx_tokens * 3.6)
    hay = raw[:target_chars]
    end = hay.rfind(". ")
    if end > 0:
        hay = hay[:end + 1]
    needle = NEEDLES[needle_idx]
    desired = len(hay) // 2  # depth=0.5 — most cells use this
    sb = hay.rfind(". ", 0, max(desired, 2))
    sb = 0 if sb < 0 else sb + 2
    return hay[:sb] + needle + " " + hay[sb:]


def compute_proxies(response: str, haystack: str) -> dict:
    """Compute copy_score and novel_score for one trial."""
    rgrams = ngrams(response, n=4)
    if not rgrams:
        return {"copy_score": None, "novel_score": None, "n_resp_ngrams": 0}
    hgrams = ngrams(haystack, n=4)
    overlap = rgrams & hgrams
    copy = len(overlap) / len(rgrams)
    novel = 1.0 - copy
    return {
        "copy_score": copy,
        "novel_score": novel,
        "n_resp_ngrams": len(rgrams),
        "n_overlap": len(overlap),
    }


def load_trials():
    """Load all trials with the response text from every CSV."""
    trials = []
    for csv_path in CSVS:
        if not csv_path.exists():
            print(f"WARN: missing {csv_path}", file=sys.stderr)
            continue
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                trials.append({
                    "source":  csv_path.name,
                    "method":  r["method"],
                    "context": int(r["context"]),
                    "depth":   float(r["depth"]),
                    "needle":  int(r["needle_idx"]),
                    "pass":    int(r["pass"]),
                    "response": r["response"],
                })
    return trials


def main():
    trials = load_trials()
    print(f"Loaded {len(trials)} trials from {len(CSVS)} CSV files\n")

    # Per-trial proxy
    rows = []
    for t in trials:
        # Bypass cells that aren't part of the cliff cells we care about.
        haystack = reconstruct_haystack(t["context"], t["needle"])
        proxies = compute_proxies(t["response"], haystack)
        if proxies["copy_score"] is None:
            continue
        rows.append({**t, **proxies})

    # Group by (model_proxy, ctx, pass) — the "model_proxy" is the source CSV
    # (we don't have model name in CSV, but the source file tells us).
    def model_of(source):
        if "043236" in source:
            return "1B Q8"
        return "3B Q4"

    # Aggregate
    agg = defaultdict(lambda: {"copy": [], "novel": [], "n": 0})
    for r in rows:
        key = (model_of(r["source"]), r["context"], r["pass"])
        agg[key]["copy"].append(r["copy_score"])
        agg[key]["novel"].append(r["novel_score"])
        agg[key]["n"] += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("=" * 78)
    print(f"{'model':<8} {'ctx':>6} {'pass':>4} {'n':>4} "
          f"{'copy_score':>12} {'novel_score':>14}")
    print("-" * 78)
    for key in sorted(agg.keys()):
        model, ctx, ps = key
        d = agg[key]
        print(f"{model:<8} {ctx:>6} {ps:>4} {d['n']:>4} "
              f"{mean(d['copy']):>12.3f} {mean(d['novel']):>14.3f}")
    print("=" * 78)

    # Hypothesis test: is mean(copy | pass=0, cliff) statistically lower
    # than mean(copy | pass=1, pre-cliff)?
    print("\n--- Hypothesis test ---\n")
    print("ReDeEP signature for hallucination:")
    print("  External Context Score (= copy proxy)  : LOW")
    print("  Parametric Knowledge Score (= novel)   : HIGH")
    print()

    # Pool all PASS trials vs all FAIL trials, ignoring model/ctx
    pass_copy  = [r["copy_score"]  for r in rows if r["pass"] == 1]
    fail_copy  = [r["copy_score"]  for r in rows if r["pass"] == 0]
    pass_novel = [r["novel_score"] for r in rows if r["pass"] == 1]
    fail_novel = [r["novel_score"] for r in rows if r["pass"] == 0]

    if pass_copy and fail_copy:
        delta_copy  = mean(pass_copy)  - mean(fail_copy)
        delta_novel = mean(pass_novel) - mean(fail_novel)
        print(f"PASS trials (n={len(pass_copy)}):  copy={mean(pass_copy):.3f}  novel={mean(pass_novel):.3f}")
        print(f"FAIL trials (n={len(fail_copy)}):  copy={mean(fail_copy):.3f}  novel={mean(fail_novel):.3f}")
        print()
        print(f"Δ copy  (PASS - FAIL) = {delta_copy:+.3f}  "
              f"({'PASS copies more' if delta_copy > 0 else 'FAIL copies more'})")
        print(f"Δ novel (PASS - FAIL) = {delta_novel:+.3f}  "
              f"({'PASS invents more' if delta_novel > 0 else 'FAIL invents more'})")
        print()
        # ReDeEP says: hallucination → low External Context (copy) + high Parametric (novel)
        # In our data: PASS = retrieves, FAIL = hallucinates
        # Hypothesis: PASS has HIGH copy and LOW novel; FAIL has LOW copy and HIGH novel
        if delta_copy > 0 and delta_novel < 0:
            print("✅ HYPOTHESIS SUPPORTED at surface level:")
            print("   PASS responses have higher copy score (use the document more)")
            print("   FAIL responses have higher novel score (invent from parametric knowledge)")
            print("   This matches ReDeEP's RAG hallucination signature.")
        elif delta_copy < 0 and delta_novel > 0:
            print("❌ HYPOTHESIS REJECTED at surface level:")
            print("   FAIL responses copy MORE than PASS — opposite of ReDeEP signature.")
            print("   The cliff mechanism is fundamentally different from RAG hallucination.")
        else:
            print("⚠️  AMBIGUOUS at surface level:")
            print(f"   Direction is mixed. Need mechanistic analysis to disambiguate.")

    # Save raw data for paper
    out = Path(REPO / "bench/results/niah/redeep_proxy.json")
    with open(out, "w") as f:
        json.dump({
            "trials": rows,
            "summary": {
                "n_total": len(rows),
                "n_pass":  len([r for r in rows if r["pass"] == 1]),
                "n_fail":  len([r for r in rows if r["pass"] == 0]),
                "pass_mean_copy":  mean(pass_copy)  if pass_copy  else None,
                "pass_mean_novel": mean(pass_novel) if pass_novel else None,
                "fail_mean_copy":  mean(fail_copy)  if fail_copy  else None,
                "fail_mean_novel": mean(fail_novel) if fail_novel else None,
            }
        }, f, indent=2)
    print(f"\nSaved per-trial data to {out}")


if __name__ == "__main__":
    main()
