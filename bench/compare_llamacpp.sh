#!/bin/bash
# =============================================================================
# compare_llamacpp.sh — TurboQuant vs llama.cpp KV cache quantization benchmark
# =============================================================================
#
# This script measures TurboQuant's KV cache compression and documents
# the equivalent llama.cpp commands for fair side-by-side comparison.
#
# Usage:
#   bash bench/compare_llamacpp.sh <model.gguf> [threads]
#
# Example:
#   bash bench/compare_llamacpp.sh models/SmolLM2-1.7B-Instruct-Q8_0.gguf 6
#
# What it measures (TurboQuant -- actually runs):
#   - Perplexity (teacher-forced on fixed 1095-word test text)
#   - KV cache memory per token
#   - Generation speed (tok/s)
#
# What it documents (llama.cpp -- commands printed, not executed):
#   - Equivalent llama.cpp commands with --cache-type-k/--cache-type-v flags
#   - Expected memory usage based on llama.cpp's quantization formats
#
# IMPORTANT: All measurements use the SAME model, SAME test text, SAME hardware.
# The only variable is the KV cache quantization method.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TQ_RUN="$PROJECT_DIR/build/tq_run"
PPL_TEXT="$SCRIPT_DIR/data/ppl_test_1k.txt"
RESULTS_DIR="$SCRIPT_DIR/compare_results"

MODEL="${1:?Usage: bash bench/compare_llamacpp.sh <model.gguf> [threads]}"
THREADS="${2:-6}"

# ---------------------------------------------------------------------------
# Validate
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
    exit 1
fi

mkdir -p "$RESULTS_DIR"

MODEL_NAME=$(basename "$MODEL")
DATE_STR=$(date +%Y-%m-%d_%H%M%S)
HOSTNAME_STR=$(hostname -s 2>/dev/null || echo "unknown")
CSV_OUT="$RESULTS_DIR/comparison_${DATE_STR}.csv"

echo ""
echo "================================================================"
echo "  TurboQuant vs llama.cpp KV Cache Quantization Comparison"
echo "================================================================"
echo ""
echo "  Model:    $MODEL_NAME"
echo "  Threads:  $THREADS"
echo "  Host:     $HOSTNAME_STR"
echo "  Date:     $DATE_STR"
echo "  Text:     ppl_test_1k.txt (1095 words, ~1400 tokens)"
echo ""
echo "================================================================"

# =========================================================================
# SECTION 1: TurboQuant measurements (actually executed)
# =========================================================================

echo ""
echo "================================================================"
echo "  SECTION 1: TurboQuant Measurements (live)"
echo "================================================================"
echo ""

# Configs: "label kv_flag v_flag bits_k bits_v"
TQ_CONFIGS=(
    "TQ:uniform_4b(K)+FP16(V)    uniform_4b  fp16  4.0  16.0"
    "TQ:turbo_1b(K)+FP16(V)      turbo_kv_1b fp16  1.0  16.0"
    "TQ:turbo_1b(K)+Q4(V)        turbo_kv_1b q4    1.0  4.0"
    "TQ:turbo_3b(K)+FP16(V)      turbo_kv_3b fp16  3.0  16.0"
    "TQ:turbo_3b(K)+Q4(V)        turbo_kv_3b q4    3.0  4.0"
)

# Collect results into arrays
declare -a R_LABEL R_PPL R_NLL R_TOKS R_KV_PER_TOK R_COMPRESS R_SAVED

run_tq_config() {
    local idx=$1
    local config_line="${TQ_CONFIGS[$idx]}"
    local label=$(echo "$config_line" | awk '{print $1}')
    local kv_type=$(echo "$config_line" | awk '{print $2}')
    local v_quant=$(echo "$config_line" | awk '{print $3}')
    local bits_k=$(echo "$config_line" | awk '{print $4}')
    local bits_v=$(echo "$config_line" | awk '{print $5}')

    echo "  Running: $label ..."

    # --- PPL measurement ---
    local ppl_cmd="$TQ_RUN $MODEL --ppl $PPL_TEXT -j $THREADS -k $kv_type"
    if [ "$v_quant" != "fp16" ]; then
        ppl_cmd="$ppl_cmd -v $v_quant"
    fi
    local ppl_output
    ppl_output=$($ppl_cmd 2>&1) || true

    local ppl nll tok_s tokens
    ppl=$(echo "$ppl_output" | grep "^PPL_CSV:" | cut -d, -f3)
    nll=$(echo "$ppl_output" | grep "^PPL_CSV:" | cut -d, -f2)
    tokens=$(echo "$ppl_output" | grep "^PPL_CSV:" | cut -d, -f1 | sed 's/PPL_CSV://')
    tok_s=$(echo "$ppl_output" | grep "tok/s" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | grep -o '[0-9]*\.[0-9]*')

    # Fallback
    if [ -z "$ppl" ]; then
        ppl=$(echo "$ppl_output" | grep "Perplexity:" | grep -o '[0-9]*\.[0-9]*')
        nll=$(echo "$ppl_output" | grep "Avg NLL:" | grep -o '[0-9]*\.[0-9]*')
    fi

    # --- Memory measurement (generate 200 tokens to get meaningful KV stats) ---
    local mem_cmd="$TQ_RUN $MODEL -p 'The quick brown fox jumps over the lazy dog and continues walking through the forest path.' -n 200 -T 0.0 -j $THREADS -k $kv_type -M"
    if [ "$v_quant" != "fp16" ]; then
        mem_cmd="$mem_cmd -v $v_quant"
    fi
    local mem_output
    mem_output=$($mem_cmd 2>&1) || true

    local kv_per_tok compress_ratio mem_saved
    kv_per_tok=$(echo "$mem_output" | grep "Per-token K+V total:" | grep -o '[0-9]*\.[0-9]* KB')
    compress_ratio=$(echo "$mem_output" | grep "Compression ratio:" | grep -o '[0-9]*\.[0-9]*x')
    mem_saved=$(echo "$mem_output" | grep "Memory saved:" | grep -o '[0-9]*\.[0-9]* MB')

    # Store results
    R_LABEL[$idx]="$label"
    R_PPL[$idx]="${ppl:-N/A}"
    R_NLL[$idx]="${nll:-N/A}"
    R_TOKS[$idx]="${tok_s:-N/A}"
    R_KV_PER_TOK[$idx]="${kv_per_tok:-N/A}"
    R_COMPRESS[$idx]="${compress_ratio:-N/A}"
    R_SAVED[$idx]="${mem_saved:-N/A}"
}

for i in "${!TQ_CONFIGS[@]}"; do
    run_tq_config "$i"
done

# ---------------------------------------------------------------------------
# Print TurboQuant results table
# ---------------------------------------------------------------------------
echo ""
echo "  TurboQuant Results:"
echo "  -----------------------------------------------------------------------"
printf "  %-30s %8s %8s %10s %12s %8s\n" \
    "Config" "PPL" "NLL" "tok/s" "KV/tok" "Ratio"
printf "  %-30s %8s %8s %10s %12s %8s\n" \
    "------" "---" "---" "-----" "------" "-----"

for i in "${!TQ_CONFIGS[@]}"; do
    printf "  %-30s %8s %8s %10s %12s %8s\n" \
        "${R_LABEL[$i]}" "${R_PPL[$i]}" "${R_NLL[$i]}" \
        "${R_TOKS[$i]}" "${R_KV_PER_TOK[$i]}" "${R_COMPRESS[$i]}"
done

# =========================================================================
# SECTION 2: llama.cpp equivalent commands (documented, not executed)
# =========================================================================

echo ""
echo "================================================================"
echo "  SECTION 2: llama.cpp Equivalent Commands (reference)"
echo "================================================================"
echo ""
echo "  These commands are NOT executed by this script. They document"
echo "  the equivalent llama.cpp invocations for fair comparison."
echo "  Run them separately with a llama.cpp build to get comparable numbers."
echo ""
echo "  Prerequisites:"
echo "    cd /path/to/llama.cpp"
echo "    cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(nproc)"
echo ""

# Create a temporary PPL file path placeholder
LLAMACPP_PPL_TEXT="bench/data/ppl_test_1k.txt"

cat << 'DOCEOF'
  -----------------------------------------------------------------------
  Config                        llama.cpp command
  -----------------------------------------------------------------------

  1. Baseline (FP16 KV cache — no quantization):

     ./build/bin/llama-perplexity \
       -m MODEL.gguf \
       -f bench/data/ppl_test_1k.txt \
       --cache-type-k f16 \
       --cache-type-v f16 \
       -t THREADS

     Memory: 16 bits/value for K, 16 bits/value for V
     Per-token KV = 2 * n_layers * n_kv_heads * head_dim * 2 bytes

  2. Q8_0 K cache (8-bit quantized keys):

     ./build/bin/llama-perplexity \
       -m MODEL.gguf \
       -f bench/data/ppl_test_1k.txt \
       --cache-type-k q8_0 \
       --cache-type-v f16 \
       -t THREADS

     Memory: 8.5 bits/value for K (q8_0 has scale overhead), 16 bits/value for V
     Expected: near-lossless, PPL increase < 0.1

  3. Q4_0 K cache (4-bit quantized keys):

     ./build/bin/llama-perplexity \
       -m MODEL.gguf \
       -f bench/data/ppl_test_1k.txt \
       --cache-type-k q4_0 \
       --cache-type-v f16 \
       -t THREADS

     Memory: 4.5 bits/value for K (q4_0 has scale overhead), 16 bits/value for V
     Expected: small PPL increase, typically < 0.5

  4. Q4_0 K + Q4_0 V (4-bit K and V):

     ./build/bin/llama-perplexity \
       -m MODEL.gguf \
       -f bench/data/ppl_test_1k.txt \
       --cache-type-k q4_0 \
       --cache-type-v q4_0 \
       -t THREADS

     Memory: 4.5 bits/value for both K and V
     Expected: moderate PPL increase

  For generation speed measurement:

     ./build/bin/llama-cli \
       -m MODEL.gguf \
       -p "The quick brown fox" \
       -n 200 \
       --cache-type-k {f16|q8_0|q4_0} \
       --cache-type-v {f16|q4_0} \
       -t THREADS \
       --temp 0

  -----------------------------------------------------------------------
DOCEOF

# =========================================================================
# SECTION 3: Theoretical comparison table
# =========================================================================

echo ""
echo "================================================================"
echo "  SECTION 3: Side-by-Side Comparison"
echo "================================================================"
echo ""
echo "  Key comparison points (same model, same text):"
echo ""
echo "  -----------------------------------------------------------------------"
printf "  %-32s %6s %6s %10s %s\n" \
    "Method" "K bit" "V bit" "KV/tok" "Notes"
printf "  %-32s %6s %6s %10s %s\n" \
    "------" "-----" "-----" "------" "-----"
printf "  %-32s %6s %6s %10s %s\n" \
    "llama.cpp f16/f16 (baseline)" "16" "16" "~192 KB*" "No compression"
printf "  %-32s %6s %6s %10s %s\n" \
    "llama.cpp q8_0/f16" "8.5" "16" "~150 KB*" "Near-lossless K"
printf "  %-32s %6s %6s %10s %s\n" \
    "llama.cpp q4_0/f16" "4.5" "16" "~126 KB*" "4-bit uniform K"
printf "  %-32s %6s %6s %10s %s\n" \
    "llama.cpp q4_0/q4_0" "4.5" "4.5" "~55 KB*" "Both quantized"

echo "  -----------------------------------------------------------------------"

# Now fill in the actual TurboQuant measurements
for i in "${!TQ_CONFIGS[@]}"; do
    local_label="${R_LABEL[$i]}"
    local_bits_k=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $4}')
    local_bits_v=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $5}')
    local_kv="${R_KV_PER_TOK[$i]}"
    local_ppl="${R_PPL[$i]}"
    printf "  %-32s %6s %6s %10s %s\n" \
        "$local_label" "$local_bits_k" "$local_bits_v" "$local_kv" "PPL=$local_ppl"
done

echo "  -----------------------------------------------------------------------"
echo ""
echo "  * llama.cpp KV/tok estimates assume: 24 layers, 32 kv_heads, head_dim=64"
echo "    Formula: n_layers * n_kv_heads * head_dim * (bits_k + bits_v) / 8 bytes"
echo "    Actual values depend on model architecture; run llama.cpp to confirm."
echo ""

# =========================================================================
# SECTION 4: Key insights
# =========================================================================

echo "================================================================"
echo "  SECTION 4: Key Comparison Insights"
echo "================================================================"
echo ""
echo "  What TurboQuant offers vs llama.cpp KV quantization:"
echo ""
echo "  1. LOWER BIT RATES: TurboQuant achieves 1-bit and 3-bit K cache"
echo "     quantization using PolarQuant + QJL algorithms. llama.cpp's"
echo "     lowest is q4_0 (4.5 effective bits)."
echo ""
echo "  2. DIFFERENT ALGORITHMS: llama.cpp uses block-wise min-max (uniform)"
echo "     quantization. TurboQuant uses:"
echo "     - PolarQuant: exploits angular structure of attention keys"
echo "     - QJL: Johnson-Lindenstrauss sign hashing for 1-bit keys"
echo "     - TurboQuant: progressive residual (Polar 2b + QJL 1b = 3b)"
echo ""
echo "  3. QUALITY AT LOW BITS: The critical comparison is at the low end:"
echo "     - TurboQuant 3-bit K vs llama.cpp 4-bit K (q4_0)"
echo "     - If TurboQuant 3b matches or beats llama.cpp 4b in PPL,"
echo "       that is 25% more compression at equal quality."
echo ""
echo "  4. EXTREME COMPRESSION: TurboQuant 1-bit K + Q4 V achieves"
echo "     approximately 5x total KV compression. No llama.cpp equivalent"
echo "     exists at this bit rate."
echo ""

# =========================================================================
# SECTION 5: CSV output
# =========================================================================

echo "date,model,method,kv_type,v_quant,bits_k,bits_v,ppl,nll,tok_s,kv_per_tok,compress_ratio" > "$CSV_OUT"

for i in "${!TQ_CONFIGS[@]}"; do
    local_label="${R_LABEL[$i]}"
    local_kv_type=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $2}')
    local_v_quant=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $3}')
    local_bits_k=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $4}')
    local_bits_v=$(echo "${TQ_CONFIGS[$i]}" | awk '{print $5}')
    echo "$DATE_STR,$MODEL_NAME,turboquant,$local_kv_type,$local_v_quant,$local_bits_k,$local_bits_v,${R_PPL[$i]},${R_NLL[$i]},${R_TOKS[$i]},${R_KV_PER_TOK[$i]},${R_COMPRESS[$i]}" >> "$CSV_OUT"
done

# Add llama.cpp reference rows (no measurements, just theoretical)
echo "$DATE_STR,$MODEL_NAME,llamacpp,f16,f16,16,16,---,---,---,~192KB,1.00x" >> "$CSV_OUT"
echo "$DATE_STR,$MODEL_NAME,llamacpp,q8_0,f16,8.5,16,---,---,---,~150KB,1.28x" >> "$CSV_OUT"
echo "$DATE_STR,$MODEL_NAME,llamacpp,q4_0,f16,4.5,16,---,---,---,~126KB,1.52x" >> "$CSV_OUT"
echo "$DATE_STR,$MODEL_NAME,llamacpp,q4_0,q4_0,4.5,4.5,---,---,---,~55KB,3.49x" >> "$CSV_OUT"

echo "================================================================"
echo "  Results saved to: $CSV_OUT"
echo "================================================================"
echo ""
echo "  To complete the comparison, build llama.cpp and run the"
echo "  commands from Section 2 on the same machine with the same model."
echo "  Then paste the llama.cpp PPL numbers alongside TurboQuant's"
echo "  for a fair apples-to-apples comparison."
echo ""
echo "  Quick validation command:"
echo "    diff <(bash bench/ppl_standard.sh $MODEL) <(bash bench/ppl_standard.sh $MODEL)"
echo "  (should show identical results for reproducibility check)"
echo ""
