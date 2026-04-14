#!/bin/bash
# Multi-model regression test — verify all supported models produce coherent output.
#
# Usage: bash scripts/test_models.sh [models_dir]
#
# Exit codes:
#   0 — all expected models produce correct output
#   1 — at least one expected model failed
#
# Models are categorized into three tiers:
#   STRICT:  must produce exact substring match (high-confidence cases)
#   COHERENT: must produce non-garbage text (model quality varies)
#   SKIP:     known limitations (too slow, model too small, etc.)

set -u
MODELS_DIR="${1:-models}"
QUANT_BIN="./build/quant"
PASS=0
FAIL=0
SKIP=0

if [[ ! -x "$QUANT_BIN" ]]; then
    echo "ERROR: $QUANT_BIN not built. Run cmake --build build first." >&2
    exit 1
fi

run_test() {
    local model="$1"
    local prompt="$2"
    local expected="$3"
    local tier="$4"
    local extra_env="${5:-}"

    if [[ ! -f "$MODELS_DIR/$model" ]]; then
        printf "  %-50s [SKIP] not found\n" "$model"
        SKIP=$((SKIP + 1))
        return
    fi

    local out
    # Capture full output (replace newlines with space) — avoids missing
    # output when first line is empty (newline-prefixed generation).
    out=$(env $extra_env "$QUANT_BIN" "$MODELS_DIR/$model" -p "$prompt" -n 10 -T 0 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')

    case "$tier" in
        STRICT)
            if [[ "$out" == *"$expected"* ]]; then
                printf "  %-50s [PASS] '%s'\n" "$model" "${out:0:50}"
                PASS=$((PASS + 1))
            else
                printf "  %-50s [FAIL] expected '%s' got '%s'\n" "$model" "$expected" "${out:0:50}"
                FAIL=$((FAIL + 1))
            fi
            ;;
        COHERENT)
            # Just check non-empty and not all garbage chars
            local printable
            printable=$(echo "$out" | tr -cd '[:print:][:space:]' | wc -c)
            local total
            total=$(echo -n "$out" | wc -c)
            if [[ -n "$out" && "$total" -gt 0 && "$printable" -gt $((total / 2)) ]]; then
                printf "  %-50s [PASS] '%s'\n" "$model" "${out:0:50}"
                PASS=$((PASS + 1))
            else
                printf "  %-50s [FAIL] garbage/empty: '%s'\n" "$model" "${out:0:50}"
                FAIL=$((FAIL + 1))
            fi
            ;;
    esac
}

echo "=== quant.cpp Multi-Model Regression Test ==="
echo "Models dir: $MODELS_DIR"
echo ""

echo "--- STRICT tier (must produce expected substring) ---"
run_test "Phi-3.5-mini-instruct-Q8_0.gguf"     "2+2=" "4" STRICT "TQ_NO_METAL=1"
run_test "Phi-3.5-mini-instruct-Q4_K_M.gguf"   "2+2=" "4" STRICT "TQ_NO_METAL=1"
run_test "gemma-4-e2b-it-Q8_0.gguf"            "2+2=" "4" STRICT "TQ_NO_METAL=1 TQ_NO_Q4=1"
# Note: Llama 3.1 8B raw "2+2=" is borderline — FP32 KV gives "5: The Mathematics..."
# and turbo_kv_4b with k128 highres matches FP32. Use COHERENT tier for this model.
run_test "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "The capital of France is" "" COHERENT "TQ_NO_METAL=1"

echo ""
echo "--- COHERENT tier (must produce non-garbage text) ---"
run_test "Llama-3.2-1B-Instruct-Q8_0.gguf"     "Hello" ""  COHERENT "TQ_NO_METAL=1"
run_test "Llama-3.2-3B-Instruct-Q8_0.gguf"     "Hello" ""  COHERENT "TQ_NO_METAL=1"
run_test "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"   "Hello" ""  COHERENT "TQ_NO_METAL=1"

echo ""
echo "--- Summary ---"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
