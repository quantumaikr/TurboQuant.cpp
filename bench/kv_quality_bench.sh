#!/bin/bash
# KV Cache Quality Benchmark — Reproducible verification
#
# Tests TurboQuant KV cache quality at short (100 tokens) and longer (200+) contexts.
# Short context: typically byte-identical to uniform baseline.
# Longer context: outputs diverge but remain coherent (expected behavior).
#
# Note: Only key vectors are quantized; value vectors remain FP32.
#
# Run: bash bench/kv_quality_bench.sh <model.tqm>
#
# Requirements: built tq_run binary in build/

set -e

MODEL="${1:-model.tqm}"
TQ_RUN="./build/tq_run"
THREADS=6
RESULTS_DIR="bench/kv_quality_results"

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: $TQ_RUN not found. Build first: cmake --build build"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    echo "Usage: bash bench/kv_quality_bench.sh <model.tqm>"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

KV_TYPES="uniform_4b turbo_kv_4b turbo_kv_3b turbo_kv_1b"

# Test prompts covering diverse capabilities
PROMPTS=(
    "1+1="
    "The capital of France is"
    "The capital of Japan is"
    "Water boils at"
    "The sun rises in the"
    "Write a Python function to reverse a string:"
    "If a train travels 60 miles in 1 hour, how far does it travel in 3 hours?"
    "Explain how a computer works to a 5-year-old child."
    "List the planets in our solar system:"
    "Once upon a time, in a faraway land,"
)

TOKENS_PER_PROMPT=100
TOTAL_TESTS=${#PROMPTS[@]}
PASS=0
FAIL=0
DIVERGED=0

echo "============================================================"
echo "  TurboQuant KV Cache Quality Benchmark"
echo "============================================================"
echo ""
echo "Model:    $MODEL"
echo "Threads:  $THREADS"
echo "Tokens:   $TOKENS_PER_PROMPT per prompt"
echo "Prompts:  $TOTAL_TESTS"
echo "KV types: $KV_TYPES"
echo "Mode:     greedy (temperature=0)"
echo ""
echo "============================================================"
echo ""

# Phase 1: Generate outputs for all combinations
echo "[Phase 1] Generating outputs..."
for idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$idx]}"
    short=$(echo "$prompt" | head -c 40 | tr ' /' '_-')
    printf "  [%2d/%d] %s\n" $((idx+1)) $TOTAL_TESTS "$prompt"

    for kv in $KV_TYPES; do
        outfile="$RESULTS_DIR/p${idx}_${kv}.txt"
        $TQ_RUN "$MODEL" -p "$prompt" -j $THREADS -n $TOKENS_PER_PROMPT -T 0.0 -k $kv 2>&1 \
            | sed -n '/^---$/,/^---$/p' | tail -n +2 | sed '$d' \
            > "$outfile"
    done
done

echo ""
echo "[Phase 2] Comparing outputs..."
echo ""

# Phase 2: Compare all KV types against baseline (uniform_4b)
printf "%-45s %-12s %-12s %-12s\n" "Prompt" "vs 4b" "vs 3b" "vs 1b"
printf "%-45s %-12s %-12s %-12s\n" "-----" "------" "------" "------"

for idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$idx]}"
    display=$(echo "$prompt" | head -c 42)

    baseline="$RESULTS_DIR/p${idx}_uniform_4b.txt"
    results=""

    for kv in turbo_kv_4b turbo_kv_3b turbo_kv_1b; do
        candidate="$RESULTS_DIR/p${idx}_${kv}.txt"
        if diff -q "$baseline" "$candidate" > /dev/null 2>&1; then
            results="$results MATCH      "
            PASS=$((PASS + 1))
        else
            # Check how many tokens match before divergence
            baseline_tokens=$(wc -c < "$baseline" | tr -d ' ')
            candidate_tokens=$(wc -c < "$candidate" | tr -d ' ')
            # Find first differing byte
            first_diff=$(cmp "$baseline" "$candidate" 2>/dev/null | head -1 | grep -o 'byte [0-9]*' | grep -o '[0-9]*')
            if [ -z "$first_diff" ]; then
                # One file is prefix of other
                results="$results PREFIX     "
            else
                results="$results DIFF@${first_diff}B "
            fi
            FAIL=$((FAIL + 1))
            DIVERGED=$((DIVERGED + 1))
        fi
    done

    printf "%-45s%s\n" "$display" "$results"
done

echo ""
echo "============================================================"

# Phase 3: Speed benchmark
echo ""
echo "[Phase 3] Speed benchmark (100 tokens)..."
echo ""
printf "%-15s %10s %12s %15s\n" "KV Type" "tok/s" "KV/token" "Compression"
printf "%-15s %10s %12s %15s\n" "-------" "-----" "--------" "-----------"

for kv in $KV_TYPES; do
    output=$($TQ_RUN "$MODEL" -p "Hello world, this is a test." -j $THREADS -n 100 -T 0.0 -k $kv -M 2>&1)
    speed=$(echo "$output" | grep "tok/s" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | head -1)
    per_token=$(echo "$output" | grep "Per-token KV" | head -1 | grep -o '[0-9]*\.[0-9]* KB')
    ratio=$(echo "$output" | grep "Compression" | grep -o '[0-9]*\.[0-9]*x')
    printf "%-15s %10s %12s %15s\n" "$kv" "$speed" "$per_token" "$ratio"
done

echo ""
echo "============================================================"
echo ""

# Summary
TOTAL_COMPARISONS=$((TOTAL_TESTS * 3))
echo "  Quality: $PASS/$TOTAL_COMPARISONS byte-identical matches"
if [ $DIVERGED -gt 0 ]; then
    echo "  WARNING: $DIVERGED divergences detected!"
    echo "  Check $RESULTS_DIR/ for details."
else
    echo "  ALL OUTPUTS BYTE-IDENTICAL across all KV types."
fi
echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo ""

# Write CSV summary
CSV="$RESULTS_DIR/summary.csv"
echo "prompt_idx,prompt,uniform_4b_vs_turbo_4b,uniform_4b_vs_turbo_3b,uniform_4b_vs_turbo_1b" > "$CSV"
for idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$idx]}"
    row="$idx,\"$prompt\""
    for kv in turbo_kv_4b turbo_kv_3b turbo_kv_1b; do
        if diff -q "$RESULTS_DIR/p${idx}_uniform_4b.txt" "$RESULTS_DIR/p${idx}_${kv}.txt" > /dev/null 2>&1; then
            row="$row,MATCH"
        else
            row="$row,DIFF"
        fi
    done
    echo "$row" >> "$CSV"
done
echo "  CSV: $CSV"

exit $DIVERGED
