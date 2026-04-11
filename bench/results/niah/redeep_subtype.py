#!/usr/bin/env python3
"""R2: subtype analysis of cliff failures.

R1 found that the cliff failure is OPPOSITE to ReDeEP's RAG hallucination
signature: FAIL responses copy MORE from haystack than PASS responses do.
This contradicts the hypothesis that the two failure modes share a
mechanism.

But R1 pooled all FAIL trials. The Boulter+Sarah Chen "synthesised
hallucination" example we cited in the paper is presumably a *minority*
of failures. R2 classifies failures into subtypes:

  CONTINUE: response is dominated by haystack text the model copied
            from where the assistant turn started — this is the
            "auto-complete the document" failure mode
  HEADER:   response contains wikitext markup like "= = = 2008 II ="
            — section header echo
  SYNTH:    response fuses content from the needle with content from
            the haystack subject — the Boulter+CFO style failure
  REFUSE:   response says "I don't know" or similar (rare)
  OTHER:    none of the above

We then compute the copy/novel scores PER SUBTYPE so we can see whether
the SYNTH subtype matches the ReDeEP signature even if the pooled
average doesn't.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
PROXY = REPO / "bench/results/niah/redeep_proxy.json"
WIKITEXT = REPO / "bench/data/wikitext2_test.txt"

NEEDLE_KEYWORDS = {
    0: ["sarah", "chen", "northwind", "logistics", "financial officer", "cfo"],
    1: ["november", "san francisco", "aurora", "launch"],
    2: ["eight thousand", "8000", "helios", "reactor", "cooling tank"],
}

NEEDLE_SUBJECT_KEYWORDS = {
    # Words from the needle's *invented* entity (not from haystack)
    0: {"sarah", "chen", "northwind", "logistics"},
    1: {"aurora", "northwind"},
    2: {"helios", "northwind"},
}

WIKI_SUBJECT_KEYWORDS = {
    "boulter", "robert boulter", "whishaw", "doctors", "judge john deed",
    "the bill", "philip ridley", "mercury fur", "casualty", "kieron",
    "fletcher", "donkey punch", "daylight robbery", "paris leonti",
    "olly blackburn", "long firm", "mark strong", "derek jacobi",
}

HEADER_PATTERN = re.compile(r"= ?= ?=|^\s*=\s*[A-Z]|<\|reserved|<\|begin")
REFUSE_PATTERN = re.compile(
    r"\b(don't know|no answer|cannot|not (mention|provide|find)|"
    r"unable to|i'm sorry|insufficient)", re.I
)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def classify_failure(response: str, needle_idx: int) -> str:
    """Classify a failed response into a subtype."""
    r_norm = normalize(response)
    r_words = set(r_norm.split())

    # Subtype: REFUSE — model said it doesn't know
    if REFUSE_PATTERN.search(response):
        return "REFUSE"

    # Subtype: HEADER — wikitext markup
    if HEADER_PATTERN.search(response):
        return "HEADER"

    # Subtype: SYNTH — fuses needle subject with wiki subject
    needle_kws = NEEDLE_SUBJECT_KEYWORDS[needle_idx]
    has_needle_subject = any(kw in r_words for kw in needle_kws)
    has_wiki_subject = any(kw in r_norm for kw in WIKI_SUBJECT_KEYWORDS)
    if has_needle_subject and has_wiki_subject:
        return "SYNTH"

    # Subtype: CONTINUE — dominated by wiki content (subject present, needle absent)
    if has_wiki_subject and not has_needle_subject:
        return "CONTINUE"

    return "OTHER"


def main():
    with open(PROXY) as f:
        data = json.load(f)
    rows = data["trials"]

    failures = [r for r in rows if r["pass"] == 0]
    print(f"Total failures to classify: {len(failures)}\n")

    # Classify
    by_subtype = defaultdict(list)
    for r in failures:
        sub = classify_failure(r["response"], r["needle"])
        r["subtype"] = sub
        by_subtype[sub].append(r)

    # Per-subtype copy/novel mean
    print("=" * 78)
    print(f"{'subtype':<10} {'n':>4} {'copy':>10} {'novel':>10} "
          f"{'mean_ngrams':>13}")
    print("-" * 78)
    for sub in sorted(by_subtype.keys(), key=lambda k: -len(by_subtype[k])):
        rs = by_subtype[sub]
        n = len(rs)
        copy = sum(r["copy_score"] for r in rs) / n
        novel = sum(r["novel_score"] for r in rs) / n
        ngrams = sum(r["n_resp_ngrams"] for r in rs) / n
        print(f"{sub:<10} {n:>4} {copy:>10.3f} {novel:>10.3f} {ngrams:>13.1f}")
    print("=" * 78)

    # Compare PASS to each FAIL subtype
    pass_rows = [r for r in rows if r["pass"] == 1]
    pass_copy = sum(r["copy_score"] for r in pass_rows) / len(pass_rows)
    pass_novel = sum(r["novel_score"] for r in pass_rows) / len(pass_rows)
    print(f"\nPASS reference (n={len(pass_rows)}): copy={pass_copy:.3f}  novel={pass_novel:.3f}\n")

    print("--- ReDeEP signature comparison per failure subtype ---\n")
    print("ReDeEP says hallucination has LOW copy + HIGH novel relative to a")
    print("non-hallucinating reference. Reference here = PASS trials.\n")

    for sub in sorted(by_subtype.keys(), key=lambda k: -len(by_subtype[k])):
        rs = by_subtype[sub]
        if not rs:
            continue
        n = len(rs)
        copy = sum(r["copy_score"] for r in rs) / n
        novel = sum(r["novel_score"] for r in rs) / n
        d_copy = copy - pass_copy
        d_novel = novel - pass_novel
        marker = ""
        # ReDeEP signature: hallucination has lower copy + higher novel
        if d_copy < -0.05 and d_novel > 0.05:
            marker = " ← MATCHES ReDeEP RAG-hallucination signature"
        elif d_copy > 0.05 and d_novel < -0.05:
            marker = " ← OPPOSITE of ReDeEP signature (more copy, less novel)"
        print(f"  {sub:<10} (n={n:>3}):  Δcopy={d_copy:+.3f}  Δnovel={d_novel:+.3f}{marker}")
    print()

    # Show sample responses per subtype for the paper
    print("--- Sample responses per subtype (top 3 each) ---\n")
    for sub in sorted(by_subtype.keys(), key=lambda k: -len(by_subtype[k])):
        rs = by_subtype[sub][:3]
        print(f"[{sub}]  ({len(by_subtype[sub])} total)")
        for r in rs:
            resp = r["response"][:120].replace("\n", " ").strip()
            print(f"  ctx={r['context']:>4} n{r['needle']}: \"{resp}\"")
        print()

    # Save subtype-augmented data for paper
    out = REPO / "bench/results/niah/redeep_subtype.json"
    with open(out, "w") as f:
        json.dump({
            "by_subtype": {
                sub: {
                    "n": len(rs),
                    "mean_copy":  sum(r["copy_score"]  for r in rs) / len(rs),
                    "mean_novel": sum(r["novel_score"] for r in rs) / len(rs),
                    "samples": [r["response"][:200] for r in rs[:5]],
                }
                for sub, rs in by_subtype.items()
            },
            "pass_reference": {
                "n":          len(pass_rows),
                "mean_copy":  pass_copy,
                "mean_novel": pass_novel,
            }
        }, f, indent=2)
    print(f"Saved subtype data to {out}")


if __name__ == "__main__":
    main()
