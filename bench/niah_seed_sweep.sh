#!/usr/bin/env bash
# Cliff-cell seed sweep for the working memory cliff tech report.
# Runs the two cliff transition cells (1B Q8 ctx=1024 and 3B Q4 ctx=1280)
# with 5 random seeds × 3 needles × 2 methods = 60 trials per cell.
#
# Goal: confirm whether the cliff cell's mid-failure rate (1B fp32 4/9
# at ctx=1024) is statistically distinguishable from random or whether
# it's binomial noise. With 5 seeds × 3 needles = 15 samples per
# (model, ctx, method) combination we can compute proper Wilson
# confidence intervals.

set -e
export LC_ALL=C
export LANG=C

TQ=${TQ:-./build_metal/quant}
THREADS=${THREADS:-8}
OUT_DIR=bench/results/niah
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
RAW_LOG="$OUT_DIR/raw_seedsweep_${RUN_ID}.log"
RESULT_CSV="$OUT_DIR/results_seedsweep_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
echo "model,method,context,depth,needle_idx,seed,pass,response" > "$RESULT_CSV"

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

# (model, ctx) cliff cells to sample
CELL_MODELS=(
  "models/Llama-3.2-1B-Instruct-Q8_0.gguf"
  "models/Llama-3.2-3B-Instruct-Q8_0.gguf"
)
CELL_CONTEXTS=(1024 1280)
CELL_NAMES=("1B" "3B")
DEPTH=0.5
SEEDS=(42 1337 7 2024 31415)

METHOD_NAMES=("fp32" "turbo_q4_w128")
METHOD_FLAGS=("-k fp32" "-k turbo_kv_4b -v q4 --k-window 128")

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

total=$(( ${#CELL_MODELS[@]} * ${#NEEDLES[@]} * ${#SEEDS[@]} * ${#METHOD_NAMES[@]} ))
run_idx=0

echo "==> NIAH cliff-cell seed sweep"
echo "    binary: $TQ"
echo "    cells:  ${CELL_NAMES[*]}"
echo "    seeds:  ${SEEDS[*]}"
echo "    total:  $total trials"
echo "    raw:    $RAW_LOG"
echo "    csv:    $RESULT_CSV"
echo ""

for ci in "${!CELL_MODELS[@]}"; do
  model="${CELL_MODELS[$ci]}"
  ctx="${CELL_CONTEXTS[$ci]}"
  cell_name="${CELL_NAMES[$ci]}"
  cli_ctx=$(( ctx + 256 ))

  for mi in "${!METHOD_NAMES[@]}"; do
    mname="${METHOD_NAMES[$mi]}"
    mflags="${METHOD_FLAGS[$mi]}"
    for ni in "${!NEEDLES[@]}"; do
      needle="${NEEDLES[$ni]}"
      question="${QUESTIONS[$ni]}"
      keyword="${KEYWORDS[$ni]}"
      prompt=$(build_prompt "$ctx" "$needle" "$question")

      for seed in "${SEEDS[@]}"; do
        run_idx=$(( run_idx + 1 ))
        printf "[%3d/%d] %-2s %-14s ctx=%-5d needle=%d seed=%-5d  " \
          "$run_idx" "$total" "$cell_name" "$mname" "$ctx" "$ni" "$seed"

        out=$( "$TQ" "$model" -p "$prompt" -n 32 -T 0.0 -s "$seed" -j "$THREADS" \
                 --chat --ctx "$cli_ctx" $mflags 2>&1 || true )

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
          pass=0; echo "FAIL: ${resp:0:50}"
        fi

        echo "$cell_name,$mname,$ctx,$DEPTH,$ni,$seed,$pass,\"$resp_csv\"" >> "$RESULT_CSV"
        echo "===== $cell_name $mname ctx=$ctx needle=$ni seed=$seed =====" >> "$RAW_LOG"
        echo "$out" >> "$RAW_LOG"
        echo "" >> "$RAW_LOG"
      done
    done
  done
done

echo ""
echo "==> Summary by (model × method):"
for ci in "${!CELL_MODELS[@]}"; do
  cell_name="${CELL_NAMES[$ci]}"
  for mname in "${METHOD_NAMES[@]}"; do
    pass=$(awk -F, -v cn="$cell_name" -v m="$mname" 'NR>1 && $1==cn && $2==m {p+=$7; t++} END{printf "%d/%d", p, t}' "$RESULT_CSV")
    pct=$(awk -F, -v cn="$cell_name" -v m="$mname" 'NR>1 && $1==cn && $2==m {p+=$7; t++} END{if(t>0)printf "%.0f%%", 100*p/t}' "$RESULT_CSV")
    printf "  %-2s %-14s  %s  (%s)\n" "$cell_name" "$mname" "$pass" "$pct"
  done
done
