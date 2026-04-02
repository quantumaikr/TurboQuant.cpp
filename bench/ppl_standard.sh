#!/bin/bash
# =============================================================================
# ppl_standard.sh — Standardized perplexity evaluation for TurboQuant
# =============================================================================
#
# Runs perplexity evaluation on the standard test text (bench/data/ppl_test_1k.txt)
# across all KV cache quantization configurations.
#
# Usage:
#   bash bench/ppl_standard.sh <model.gguf> [threads]
#
# Example:
#   bash bench/ppl_standard.sh models/SmolLM2-1.7B-Instruct-Q8_0.gguf 6
#
# Output:
#   - Per-config PPL, NLL, tok/s
#   - CSV file at bench/data/ppl_results.csv
#   - Machine-readable PPL_CSV lines on stdout
#
# The test text is Pride and Prejudice Chapter 1 (public domain, 1095 words,
# ~1400 tokens). This provides a consistent evaluation across runs and machines.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TQ_RUN="$PROJECT_DIR/build/tq_run"
PPL_TEXT="$SCRIPT_DIR/data/ppl_test_1k.txt"
RESULTS_CSV="$SCRIPT_DIR/data/ppl_results.csv"

MODEL="${1:?Usage: bash bench/ppl_standard.sh <model.gguf> [threads]}"
THREADS="${2:-6}"

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "$TQ_RUN" ]; then
    echo "ERROR: $TQ_RUN not found. Build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi
if [ ! -f "$PPL_TEXT" ]; then
    echo "ERROR: Test text not found: $PPL_TEXT"
    echo "  Expected: bench/data/ppl_test_1k.txt"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL")
WORD_COUNT=$(wc -w < "$PPL_TEXT" | tr -d ' ')
DATE_STR=$(date +%Y-%m-%d_%H%M%S)

echo "============================================================"
echo "  TurboQuant Standardized Perplexity Evaluation"
echo "============================================================"
echo ""
echo "  Model:     $MODEL_NAME"
echo "  Threads:   $THREADS"
echo "  Test text: ppl_test_1k.txt ($WORD_COUNT words)"
echo "  Date:      $DATE_STR"
echo ""
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# KV configurations to test
# ---------------------------------------------------------------------------
# Each entry: "label kv_flag v_flag description"
CONFIGS=(
    "FP32_baseline           fp32        fp16  FP32 K + FP16 V (no KV quantization)"
    "uniform_4b_K            uniform_4b  fp16  Uniform 4-bit K + FP16 V"
    "turbo_1b_K              turbo_kv_1b fp16  TurboQuant 1-bit K + FP16 V"
    "turbo_1b_K_q4_V         turbo_kv_1b q4    TurboQuant 1-bit K + Q4 V"
    "turbo_3b_K              turbo_kv_3b fp16  TurboQuant 3-bit K + FP16 V"
    "turbo_3b_K_q4_V         turbo_kv_3b q4    TurboQuant 3-bit K + Q4 V"
    "turbo_4b_K              turbo_kv_4b fp16  TurboQuant 4-bit K + FP16 V"
    "turbo_4b_K_q4_V         turbo_kv_4b q4    TurboQuant 4-bit K + Q4 V"
)

# CSV header
echo "date,model,label,kv_type,v_quant,tokens,nll,ppl,tok_s" > "$RESULTS_CSV"

# Table header
printf "%-28s %8s %8s %10s %8s  %s\n" "Config" "PPL" "NLL" "tok/s" "tokens" "Description"
printf "%-28s %8s %8s %10s %8s  %s\n" "------" "---" "---" "-----" "------" "-----------"

for config_line in "${CONFIGS[@]}"; do
    label=$(echo "$config_line" | awk '{print $1}')
    kv_type=$(echo "$config_line" | awk '{print $2}')
    v_quant=$(echo "$config_line" | awk '{print $3}')
    desc=$(echo "$config_line" | awk '{for(i=4;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/ *$//')

    # Build command
    CMD="$TQ_RUN $MODEL --ppl $PPL_TEXT -j $THREADS -k $kv_type"
    if [ "$v_quant" != "fp16" ]; then
        CMD="$CMD -v $v_quant"
    fi

    # Run and parse
    output=$($CMD 2>&1) || true

    tokens=$(echo "$output" | grep "^PPL_CSV:" | cut -d, -f1 | sed 's/PPL_CSV://')
    nll=$(echo "$output" | grep "^PPL_CSV:" | cut -d, -f2)
    ppl=$(echo "$output" | grep "^PPL_CSV:" | cut -d, -f3)
    tok_s=$(echo "$output" | grep "tok/s" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | grep -o '[0-9]*\.[0-9]*')

    # Fallback parsing if PPL_CSV not found
    if [ -z "$ppl" ]; then
        ppl=$(echo "$output" | grep "Perplexity:" | grep -o '[0-9]*\.[0-9]*')
        nll=$(echo "$output" | grep "Avg NLL:" | grep -o '[0-9]*\.[0-9]*')
        tokens=$(echo "$output" | grep "Tokens:" | grep -o '[0-9]*' | head -1)
    fi

    # Default values if parsing failed
    ppl="${ppl:-N/A}"
    nll="${nll:-N/A}"
    tokens="${tokens:-N/A}"
    tok_s="${tok_s:-N/A}"

    printf "%-28s %8s %8s %10s %8s  %s\n" "$label" "$ppl" "$nll" "$tok_s" "$tokens" "$desc"
    echo "$DATE_STR,$MODEL_NAME,$label,$kv_type,$v_quant,$tokens,$nll,$ppl,$tok_s" >> "$RESULTS_CSV"
done

echo ""
echo "============================================================"
echo "  Results saved to: $RESULTS_CSV"
echo "============================================================"
echo ""
echo "  Interpretation:"
echo "    PPL = perplexity (lower is better, measures prediction quality)"
echo "    NLL = negative log-likelihood per token (lower is better)"
echo "    A PPL difference of <0.5 is generally negligible"
echo "    A PPL difference of <2.0 is acceptable for most applications"
echo ""
