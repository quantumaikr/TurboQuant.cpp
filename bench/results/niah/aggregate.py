#!/usr/bin/env python3
"""Aggregate NIAH CSV results into a markdown table.

Usage:
    python bench/results/niah/aggregate.py bench/results/niah/results_*.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(csv_path):
    rows = []
    # errors='replace' handles garbage bytes from model responses that
    # leaked non-UTF-8 sequences into the csv response column.
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "method":  r["method"],
                "context": int(r["context"]),
                "depth":   float(r["depth"]),
                "needle":  int(r["needle_idx"]),
                "pass":    int(r["pass"]),
            })
    return rows


def by_method(rows):
    agg = defaultdict(lambda: {"p": 0, "t": 0})
    for r in rows:
        agg[r["method"]]["p"] += r["pass"]
        agg[r["method"]]["t"] += 1
    return agg


def by_method_ctx(rows):
    agg = defaultdict(lambda: {"p": 0, "t": 0})
    for r in rows:
        key = (r["method"], r["context"])
        agg[key]["p"] += r["pass"]
        agg[key]["t"] += 1
    return agg


def by_method_depth(rows):
    agg = defaultdict(lambda: {"p": 0, "t": 0})
    for r in rows:
        key = (r["method"], r["depth"])
        agg[key]["p"] += r["pass"]
        agg[key]["t"] += 1
    return agg


def fmt(p, t):
    if t == 0:
        return "n/a"
    return f"{p}/{t} ({100*p/t:.0f}%)"


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    csv_path = sys.argv[1]
    rows = load(csv_path)
    if not rows:
        print("No rows."); sys.exit(1)

    methods = sorted({r["method"] for r in rows})
    contexts = sorted({r["context"] for r in rows})
    depths = sorted({r["depth"] for r in rows})

    print(f"# NIAH Results — `{Path(csv_path).name}`\n")
    print(f"- Methods: {', '.join(methods)}")
    print(f"- Contexts: {contexts}")
    print(f"- Depths: {depths}")
    print(f"- Total runs: {len(rows)}\n")

    # Overall
    print("## Overall accuracy\n")
    print("| Method | Score |")
    print("|---|---|")
    bym = by_method(rows)
    for m in methods:
        s = bym[m]
        print(f"| `{m}` | {fmt(s['p'], s['t'])} |")
    print()

    # Method × context
    print("## Accuracy by context length\n")
    header = "| Method | " + " | ".join(f"{c}" for c in contexts) + " |"
    sep    = "|" + "---|" * (len(contexts) + 1)
    print(header)
    print(sep)
    bymc = by_method_ctx(rows)
    for m in methods:
        cells = [f"`{m}`"]
        for c in contexts:
            s = bymc[(m, c)]
            cells.append(fmt(s["p"], s["t"]))
        print("| " + " | ".join(cells) + " |")
    print()

    # Method × depth
    print("## Accuracy by needle depth\n")
    header = "| Method | " + " | ".join(f"{d:.2f}" for d in depths) + " |"
    sep    = "|" + "---|" * (len(depths) + 1)
    print(header)
    print(sep)
    bymd = by_method_depth(rows)
    for m in methods:
        cells = [f"`{m}`"]
        for d in depths:
            s = bymd[(m, d)]
            cells.append(fmt(s["p"], s["t"]))
        print("| " + " | ".join(cells) + " |")
    print()

    # Delta vs first method (baseline)
    if len(methods) >= 2:
        baseline = methods[0]
        print(f"## Delta vs `{baseline}` baseline\n")
        bym = by_method(rows)
        b_acc = bym[baseline]["p"] / max(bym[baseline]["t"], 1)
        for m in methods:
            if m == baseline:
                continue
            acc = bym[m]["p"] / max(bym[m]["t"], 1)
            delta = (acc - b_acc) * 100
            sign = "+" if delta >= 0 else ""
            print(f"- `{m}`: **{sign}{delta:.1f} pp** vs baseline ({100*acc:.1f}% vs {100*b_acc:.1f}%)")
        print()


if __name__ == "__main__":
    main()
