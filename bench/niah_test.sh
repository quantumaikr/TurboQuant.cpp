#!/usr/bin/env bash
# Needle-in-a-Haystack benchmark for quant.cpp KV cache compression.
#
# Compares FP32 KV (baseline) vs turbo_kv_4b -v q4 --k-window 128 (6.4× compression).
# Uses common-English-word needles that survive Q4 weight visual jitter.
# Scoring: case-insensitive grep for distinctive keywords from the needle.
#
# Usage:
#   bash bench/niah_test.sh                  # default grid
#   GRID=quick bash bench/niah_test.sh       # smaller grid for fast iteration
#   GRID=full  bash bench/niah_test.sh       # full grid (slow)

set -e

# Force byte-level locale for all child processes — the model can emit
# multibyte UTF-8 sequences and the default macOS awk path will abort
# a 90-run grid with "towc: multibyte conversion failure" on the first
# non-ASCII byte. Keeping C everywhere makes response extraction robust.
export LC_ALL=C
export LANG=C

TQ=${TQ:-./build_metal/quant}
MODEL=${MODEL:-models/Llama-3.2-3B-Instruct-Q8_0.gguf}
THREADS=${THREADS:-8}
GRID=${GRID:-default}
OUT_DIR=${OUT_DIR:-bench/results/niah}
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
RAW_LOG="$OUT_DIR/raw_${RUN_ID}.log"
RESULT_CSV="$OUT_DIR/results_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"

if [ ! -x "$TQ" ]; then
  echo "ERROR: $TQ not built. Run: cmake --build build_metal -j8" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: $MODEL missing." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Grid configuration
#
# IMPORTANT: contexts here are TOKEN counts, not chars. Llama-3.2-3B-Instruct-Q8_0
# runs from this CLI default-converts weights to Q4 on the fly. Empirically the
# effective working memory of that build is ~1500 tokens — beyond that the
# chat template gets overpowered by the document continuation prior and the
# model fails to answer the question (just continues the haystack text).
# Grid sizes therefore stay within the regime where the model can actually
# retrieve, so we measure compression-vs-baseline cleanly.
# ----------------------------------------------------------------------------
# Env-var override: set NIAH_CONTEXTS / NIAH_DEPTHS (space-separated) to
# bypass the case-based grid for ad-hoc measurement runs without editing
# this file. Example:
#   NIAH_CONTEXTS="1280 1536 1792 2048" bash bench/niah_test.sh
if [ -n "${NIAH_CONTEXTS:-}" ]; then
  # shellcheck disable=SC2206
  CONTEXTS=($NIAH_CONTEXTS)
  # shellcheck disable=SC2206
  DEPTHS=(${NIAH_DEPTHS:-0.1 0.5 0.9})
else
  case "$GRID" in
    quick)
      CONTEXTS=(512 1024)
      DEPTHS=(0.1 0.5 0.9)
      ;;
    default)
      CONTEXTS=(512 1024 1536)
      DEPTHS=(0.1 0.5 0.9)
      ;;
    full)
      CONTEXTS=(512 1024 1536)
      DEPTHS=(0.1 0.25 0.5 0.75 0.9)
      ;;
    *)
      echo "Unknown GRID: $GRID" >&2; exit 1 ;;
  esac
fi

# Three needles, all common-English-word so the answer survives Q4 jitter.
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

# Methods: name|kv-flag|v-flag|extra
METHOD_NAMES=("fp32" "turbo_q4_w128")
METHOD_FLAGS=("-k fp32" "-k turbo_kv_4b -v q4 --k-window 128")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
# build_prompt CTX_TOKENS DEPTH NEEDLE QUESTION → echoes the prompt
#
# Uses real wikitext-2 text as varied haystack (synthetic repetitive filler
# triggers a "stuck in repetition loop" failure mode in 3B Q4: the model
# generates meta-text like "I'm trapped in an infinite loop of repetition"
# instead of answering the question — see bench/results/niah/findings.md).
build_prompt() {
  python3 - "$1" "$2" "$3" "$4" <<'PYEOF'
import sys
ctx_tokens = int(sys.argv[1])
depth = float(sys.argv[2])
needle = sys.argv[3]
question = sys.argv[4]

with open("bench/data/wikitext2_test.txt") as f:
  raw = f.read()

# ~4 chars per token for English wikitext, sized below ctx to leave room
# for the question + chat template + answer headroom.
target_chars = int(ctx_tokens * 3.6)
hay = raw[:target_chars]
# Trim to last full sentence so the model isn't fed a partial word.
end = hay.rfind(". ")
if end > 0:
  hay = hay[:end + 1]

# Insert needle at sentence boundary nearest the requested depth.
desired = int(len(hay) * depth)
sb = hay.rfind(". ", 0, max(desired, 2))
if sb < 0:
  sb = 0
else:
  sb += 2
hay2 = hay[:sb] + needle + " " + hay[sb:]

# Simple format that works with --chat at sub-1500-token contexts.
# The structured "Based on this document..." prefix overpowers the
# chat template at this scale and causes the model to continue the
# haystack — keep it minimal.
prompt = hay2 + "\n\nQuestion: " + question
sys.stdout.write(prompt)
PYEOF
}

# score_response RESPONSE KEYWORD → echoes 1 (pass) or 0 (fail)
score_response() {
  local resp="$1"
  local kw="$2"
  if echo "$resp" | grep -qiE "$(echo "$kw" | sed 's/\\|/|/g')"; then
    echo 1
  else
    echo 0
  fi
}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
echo "method,context,depth,needle_idx,pass,response" > "$RESULT_CSV"
echo "==> NIAH Benchmark"
echo "    binary:  $TQ"
echo "    model:   $MODEL"
echo "    grid:    $GRID  contexts=${CONTEXTS[*]}  depths=${DEPTHS[*]}"
echo "    needles: ${#NEEDLES[@]}"
echo "    methods: ${METHOD_NAMES[*]}"
echo "    raw:     $RAW_LOG"
echo "    results: $RESULT_CSV"
echo ""

total_runs=$(( ${#METHOD_NAMES[@]} * ${#CONTEXTS[@]} * ${#DEPTHS[@]} * ${#NEEDLES[@]} ))
run_idx=0

for mi in "${!METHOD_NAMES[@]}"; do
  mname="${METHOD_NAMES[$mi]}"
  mflags="${METHOD_FLAGS[$mi]}"
  for ctx in "${CONTEXTS[@]}"; do
    # Need ctx + question + answer headroom; round up to power of 2 + slack
    cli_ctx=$(( ctx + 256 ))
    for depth in "${DEPTHS[@]}"; do
      for ni in "${!NEEDLES[@]}"; do
        run_idx=$(( run_idx + 1 ))
        needle="${NEEDLES[$ni]}"
        question="${QUESTIONS[$ni]}"
        keyword="${KEYWORDS[$ni]}"

        prompt=$(build_prompt "$ctx" "$depth" "$needle" "$question")

        printf "[%3d/%d] %-14s ctx=%-5d depth=%.2f needle=%d  " \
          "$run_idx" "$total_runs" "$mname" "$ctx" "$depth" "$ni"

        # Run inference
        out=$( "$TQ" "$MODEL" -p "$prompt" -n 32 -T 0.0 -j "$THREADS" \
                 --chat --ctx "$cli_ctx" $mflags 2>&1 || true )

        # Extract response — between 1st and 2nd '---' delimiters,
        # skipping the [tokenizer] line that the CLI prints first.
        resp=$(echo "$out" | awk '
          /^---$/   { n++; next }
          n==1 && /^\[tokenizer\]/ { next }
          n==1      { print }
        ')
        if [ -z "$resp" ]; then
          resp=$(echo "$out" | tail -3 | head -1)
        fi
        # Strip newlines for CSV
        resp_csv=$(echo "$resp" | tr '\n' ' ' | sed 's/"/""/g')

        pass=$(score_response "$resp" "$keyword")
        if [ "$pass" = "1" ]; then echo "PASS"; else echo "FAIL: ${resp:0:60}"; fi

        echo "$mname,$ctx,$depth,$ni,$pass,\"$resp_csv\"" >> "$RESULT_CSV"
        echo "===== $mname ctx=$ctx depth=$depth needle=$ni =====" >> "$RAW_LOG"
        echo "$out" >> "$RAW_LOG"
        echo "" >> "$RAW_LOG"
      done
    done
  done
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "==> Results CSV: $RESULT_CSV"
echo ""
echo "==> Summary by method:"
for mname in "${METHOD_NAMES[@]}"; do
  pass=$(awk -F, -v m="$mname" 'NR>1 && $1==m {p+=$5; t++} END{printf "%d/%d", p, t}' "$RESULT_CSV")
  pct=$(awk -F, -v m="$mname" 'NR>1 && $1==m {p+=$5; t++} END{if(t>0)printf "%.1f%%", 100*p/t; else print "n/a"}' "$RESULT_CSV")
  printf "  %-16s  %s  (%s)\n" "$mname" "$pass" "$pct"
done

echo ""
echo "==> Summary by (method × context):"
printf "  %-16s" "method"
for ctx in "${CONTEXTS[@]}"; do printf " %7d" "$ctx"; done
echo ""
for mname in "${METHOD_NAMES[@]}"; do
  printf "  %-16s" "$mname"
  for ctx in "${CONTEXTS[@]}"; do
    pct=$(awk -F, -v m="$mname" -v c="$ctx" 'NR>1 && $1==m && $2==c {p+=$5; t++} END{if(t>0)printf "%5.0f%%", 100*p/t; else print "  n/a"}' "$RESULT_CSV")
    printf " %7s" "$pct"
  done
  echo ""
done
