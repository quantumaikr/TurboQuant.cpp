#!/usr/bin/env bash
# Minimal FP32-weights control experiment for the working memory cliff
# tech report. Reuses bench/niah_test.sh's prompt format and scoring but
# only runs the 6 cells that bracket the cliff transition (the full
# 36-run grid is infeasible because the FP32-weights path runs at
# ~0.02 tok/s on Metal — see TQ_NO_Q4 in tools/quant.c).
#
# Goal: measure whether the cliff location depends on weight precision.
# If 3B FP32 at ctx=1024 passes and ctx=1280 fails (matching the Q4
# default), the cliff is *independent* of weight quantization, meaning
# the ceiling is a property of the model's instruction-following
# robustness rather than its weight precision.

set -e
export LC_ALL=C
export LANG=C

TQ=${TQ:-./build_metal/quant}
MODEL=${MODEL:-models/Llama-3.2-3B-Instruct-Q8_0.gguf}
THREADS=${THREADS:-8}
OUT_DIR=bench/results/niah
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
RAW_LOG="$OUT_DIR/raw_fp32ctrl_${RUN_ID}.log"
RESULT_CSV="$OUT_DIR/results_fp32ctrl_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
echo "method,context,depth,needle_idx,pass,response" > "$RESULT_CSV"

# Same three needles as the main grid
NEEDLE_0="The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023."
QUESTION_0="Who is the chief financial officer of Northwind Logistics? Answer with the full name."
KEYWORD_0="Sarah\|Chen"

NEEDLE_1="The launch date for Project Aurora is November 14th in San Francisco."
QUESTION_1="When and where will Project Aurora launch? Answer in one sentence."
KEYWORD_1="November\|San Francisco"

NEEDLE_2="The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water."
QUESTION_2="How much distilled water does the reactor cooling tank at Helios hold?"
KEYWORD_2="eight thousand\|8000\|8,000"

NEEDLES=("$NEEDLE_0" "$NEEDLE_1" "$NEEDLE_2")
QUESTIONS=("$QUESTION_0" "$QUESTION_1" "$QUESTION_2")
KEYWORDS=("$KEYWORD_0" "$KEYWORD_1" "$KEYWORD_2")

# Just the cliff transition cells
CONTEXTS=(1024 1280)
DEPTH=0.5  # mid-document only — depth sensitivity already characterised

build_prompt() {
  local ctx_tokens="$1" needle="$2" question="$3"
  python3 - "$ctx_tokens" "$needle" "$question" <<'PYEOF'
import sys
ctx_tokens=int(sys.argv[1]); needle=sys.argv[2]; question=sys.argv[3]
with open("bench/data/wikitext2_test.txt") as f:
  raw=f.read()
target=int(ctx_tokens*3.6)
hay=raw[:target]
end=hay.rfind(". ")
if end>0: hay=hay[:end+1]
sb=hay.rfind(". ", 0, max(len(hay)//2,2))
sb = 0 if sb<0 else sb+2
h=hay[:sb]+needle+" "+hay[sb:]
sys.stdout.write(h+"\n\nQuestion: "+question)
PYEOF
}

run_idx=0
total=$(( ${#CONTEXTS[@]} * ${#NEEDLES[@]} ))

echo "==> FP32-weights control experiment"
echo "    binary:  $TQ"
echo "    model:   $MODEL"
echo "    flag:    TQ_NO_Q4=1 (loads weights as FP32)"
echo "    cells:   contexts=${CONTEXTS[*]}  depth=$DEPTH  needles=${#NEEDLES[@]}"
echo "    raw:     $RAW_LOG"
echo "    results: $RESULT_CSV"
echo ""

for ctx in "${CONTEXTS[@]}"; do
  cli_ctx=$(( ctx + 256 ))
  for ni in "${!NEEDLES[@]}"; do
    run_idx=$(( run_idx + 1 ))
    needle="${NEEDLES[$ni]}"
    question="${QUESTIONS[$ni]}"
    keyword="${KEYWORDS[$ni]}"

    prompt=$(build_prompt "$ctx" "$needle" "$question")
    printf "[%d/%d] fp32-w  ctx=%d  needle=%d  " "$run_idx" "$total" "$ctx" "$ni"

    out=$(TQ_NO_Q4=1 "$TQ" "$MODEL" -p "$prompt" -n 32 -T 0.0 -j "$THREADS" \
            --chat --ctx "$cli_ctx" -k fp32 2>&1 || true)

    resp=$(echo "$out" | awk '
      /^---$/   { n++; next }
      n==1 && /^\[tokenizer\]/ { next }
      n==1      { print }
    ' || true)
    if [ -z "$resp" ]; then resp=$(echo "$out" | tail -3 | head -1); fi

    resp_csv=$(echo "$resp" | tr '\n' ' ' | sed 's/"/""/g')
    if echo "$resp" | grep -qiE "$(echo "$keyword" | sed 's/\\|/|/g')"; then
      pass=1; echo "PASS"
    else
      pass=0; echo "FAIL: ${resp:0:60}"
    fi

    echo "fp32-weights,$ctx,$DEPTH,$ni,$pass,\"$resp_csv\"" >> "$RESULT_CSV"
    echo "===== fp32-weights ctx=$ctx needle=$ni =====" >> "$RAW_LOG"
    echo "$out" >> "$RAW_LOG"
    echo "" >> "$RAW_LOG"
  done
done

echo ""
echo "==> Summary by context:"
for ctx in "${CONTEXTS[@]}"; do
  pass=$(awk -F, -v c="$ctx" 'NR>1 && $2==c {p+=$5; t++} END{printf "%d/%d", p, t}' "$RESULT_CSV")
  pct=$(awk -F, -v c="$ctx" 'NR>1 && $2==c {p+=$5; t++} END{if(t>0)printf "%.0f%%", 100*p/t}' "$RESULT_CSV")
  printf "  ctx=%-5d  %s  (%s)\n" "$ctx" "$pass" "$pct"
done
