#!/usr/bin/env bash
# test_prefill.sh — measure prompt prefill throughput.
#
# Why: generation throughput (tok/s during decode) is what user-facing
# benchmarks usually report, but for any RAG/long-context workload the
# user waits on PREFILL — running the prompt through the model to build
# the KV cache. quant.cpp currently does prefill one token at a time
# through the same forward path as generation, so prefill ≈ gen rate
# instead of the typical ~10-100× speedup that batched matmul gives.
#
# This script makes the gap measurable. Compare the printed pp_tps
# against `llama-bench -p 512 -n 0` for the same model.
#
# Usage: bash scripts/test_prefill.sh [models_dir]

set -u
MODELS_DIR="${1:-models}"
QUANT_BIN="${QUANT_BIN:-./build/quant}"

if [[ ! -x "$QUANT_BIN" ]]; then
    echo "ERROR: $QUANT_BIN not built." >&2
    exit 1
fi

# Build a 200-token-ish prompt by repeating a known phrase.
# Phi-3.5/Qwen tokenize this at ~1 token per word.
make_prompt() {
    local n_words=$1
    local out=""
    for ((i=0; i<n_words; i++)); do
        out+="The quick brown fox jumps over the lazy dog. "
    done
    echo -n "$out"
}

bench_prefill() {
    local model="$1"
    local n_words="$2"
    local mode_label="${3:-baseline}"
    local extra_args="${4:-}"
    if [[ ! -f "$MODELS_DIR/$model" ]]; then
        printf "  %-40s %4dw  %-12s  [SKIP]\n" "$model" "$n_words" "$mode_label"
        return
    fi
    local prompt
    prompt=$(make_prompt "$n_words")
    local prompt_chars=${#prompt}

    local t0 t1 elapsed
    t0=$(date +%s.%N)
    "$QUANT_BIN" "$MODELS_DIR/$model" $extra_args -p "$prompt" -n 1 -T 0 > /dev/null 2>&1
    t1=$(date +%s.%N)
    elapsed=$(echo "$t1 - $t0" | bc -l)
    local approx_toks=$(( prompt_chars / 5 ))
    local rate=$(echo "scale=1; $approx_toks / $elapsed" | bc -l)
    printf "  %-40s %4dw  %-12s  %6.1fs  pp_tps≈%s\n" \
        "$model" "$n_words" "$mode_label" "$elapsed" "$rate"
}

echo "=== Prefill throughput (TQ_NO_METAL=1) ==="
echo "Note: pp_tps is approximate (chars/5). Compare to llama-bench -p N -n 0."
echo ""

export TQ_NO_METAL=1

# Two prompt sizes for each model: small (~50 tok) and medium (~250 tok).
# The 1000+ token sweep takes 10+ minutes per model — uncomment to run.
for model in \
    Llama-3.2-1B-Instruct-Q8_0.gguf \
    Llama-3.2-3B-Instruct-Q8_0.gguf \
    Phi-3.5-mini-instruct-Q4_K_M.gguf \
    Qwen3.5-4B-Q4_K_M.gguf; do
    bench_prefill "$model" 10   # ~50 tokens
    bench_prefill "$model" 50   # ~250 tokens
done

echo ""
echo "=== With -k fp32 (batched prefill auto-enabled, ~2-4× speedup on prefill) ==="
for model in \
    Llama-3.2-1B-Instruct-Q8_0.gguf \
    Llama-3.2-3B-Instruct-Q8_0.gguf; do
    bench_prefill "$model" 50 "-k fp32" "-k fp32"
done
