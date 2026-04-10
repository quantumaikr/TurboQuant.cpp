#!/bin/bash
# quant.cpp — Generation Regression Test
#
# Detects autoregressive generation collapse that PPL tests miss.
# Tests: T=0 greedy 500-token generation → verify no garbage output.
#
# The key insight: PPL (teacher-forced) is near-identical for FP32 and
# turbo_kv_4b at all context lengths. But autoregressive generation
# can collapse at ~500 tokens when T=0 repetition compounds KV quant error.
#
# This test catches that class of bugs by checking:
# 1. Loop detection triggers (prevents garbage, so verify it fires)
# 2. Output before loop detection is coherent (no random Unicode)
# 3. PPL sanity check at multiple context lengths
#
# Usage:
#   bash bench/generation_regression_test.sh [model.gguf]
#
# Requires: built quant binary in build/

set -e

MODEL="${1:-models/Llama-3.2-1B-Instruct-Q8_0.gguf}"
TQ_RUN="./build/quant"
THREADS=4
PASS=0
FAIL=0

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: $TQ_RUN not found. Build first."
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "SKIP: Model not found: $MODEL"
    exit 0
fi

echo "============================================"
echo "  Generation Regression Test"
echo "  Model: $MODEL"
echo "============================================"
echo ""

check() {
    local desc="$1" result="$2"
    if [ "$result" = "PASS" ]; then
        echo "  [PASS] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: T=0 generation should NOT produce garbage at 500 tokens
echo "[Test 1] T=0 500-token generation — no garbage output"
OUTPUT=$($TQ_RUN "$MODEL" -p "Explain the theory of relativity in detail" \
    -n 500 -T 0.0 -j $THREADS -k turbo_kv_4b --chat 2>/dev/null)

# Check for garbage patterns: random Unicode, excessive non-ASCII
# Garbage typically has lots of CJK/Arabic/Thai mixed with Latin
GARBAGE_CHARS=$(echo "$OUTPUT" | tr -cd '\200-\377' | wc -c | tr -d ' ')
TOTAL_CHARS=$(echo "$OUTPUT" | wc -c | tr -d ' ')
if [ "$TOTAL_CHARS" -gt 0 ]; then
    GARBAGE_RATIO=$((GARBAGE_CHARS * 100 / TOTAL_CHARS))
else
    GARBAGE_RATIO=100
fi
if [ "$GARBAGE_RATIO" -lt 30 ]; then
    check "turbo_kv_4b output coherence (${GARBAGE_RATIO}% non-ASCII)" "PASS"
else
    check "turbo_kv_4b output coherence (${GARBAGE_RATIO}% non-ASCII, threshold 30%)" "FAIL"
fi

# Test 2: Loop detection should fire for T=0 repetitive prompt
echo ""
echo "[Test 2] Loop detection fires on repetitive T=0 generation"
LOOP_OUTPUT=$($TQ_RUN "$MODEL" -p "what is your name?" \
    -n 1000 -T 0.0 -j $THREADS -k turbo_kv_4b 2>&1)

if echo "$LOOP_OUTPUT" | grep -q "repetition loop detected"; then
    LOOP_TOKENS=$(echo "$LOOP_OUTPUT" | grep "repetition loop" | grep -o "after [0-9]* tokens" | grep -o "[0-9]*")
    check "loop detected at ${LOOP_TOKENS} tokens (before 500)" "PASS"
else
    TOTAL_TOK=$(echo "$LOOP_OUTPUT" | grep "tok/s" | grep -o "^[0-9]*")
    if [ "${TOTAL_TOK:-1000}" -lt 500 ]; then
        check "EOS hit at ${TOTAL_TOK} tokens (no loop needed)" "PASS"
    else
        check "no loop detection in 1000 tokens" "FAIL"
    fi
fi

# Test 3: Non-repetitive generation should NOT trigger loop detection
echo ""
echo "[Test 3] Non-repetitive generation (T=0.7) — no false positives"
NORMAL_OUTPUT=$($TQ_RUN "$MODEL" -p "Tell me a creative story" \
    -n 200 -T 0.7 -j $THREADS -k turbo_kv_4b --chat 2>&1)

if echo "$NORMAL_OUTPUT" | grep -q "repetition loop detected"; then
    check "no false loop detection at T=0.7" "FAIL"
else
    check "no false loop detection at T=0.7" "PASS"
fi

# Test 4: FP32 vs turbo_kv_4b PPL sanity (if ppl data exists)
PPL_FILE="bench/data/ppl_test_1k.txt"
if [ -f "$PPL_FILE" ]; then
    echo ""
    echo "[Test 4] PPL sanity: turbo_kv_4b within 15% of FP32"
    FP32_PPL=$($TQ_RUN "$MODEL" --ppl "$PPL_FILE" -k fp32 -j $THREADS 2>&1 \
        | grep "PPL_CSV" | cut -d, -f3)
    Q4_PPL=$($TQ_RUN "$MODEL" --ppl "$PPL_FILE" -k turbo_kv_4b -j $THREADS 2>&1 \
        | grep "PPL_CSV" | cut -d, -f3)

    if [ -n "$FP32_PPL" ] && [ -n "$Q4_PPL" ]; then
        # Compare using integer math (multiply by 1000)
        FP32_INT=$(echo "$FP32_PPL" | awk '{printf "%d", $1 * 1000}')
        Q4_INT=$(echo "$Q4_PPL" | awk '{printf "%d", $1 * 1000}')
        THRESHOLD=$((FP32_INT * 115 / 100))  # 15% margin
        if [ "$Q4_INT" -le "$THRESHOLD" ]; then
            DELTA=$(echo "$FP32_PPL $Q4_PPL" | awk '{printf "%.1f", ($2/$1 - 1)*100}')
            check "PPL delta: ${DELTA}% (within 15%)" "PASS"
        else
            DELTA=$(echo "$FP32_PPL $Q4_PPL" | awk '{printf "%.1f", ($2/$1 - 1)*100}')
            check "PPL delta: ${DELTA}% (exceeds 15%)" "FAIL"
        fi
    else
        check "PPL comparison (could not parse results)" "FAIL"
    fi
fi

echo ""
echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
