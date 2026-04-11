#!/usr/bin/env bash
# Anchor-strengthening mitigation test for the working memory cliff.
#
# Phase 2B identified the cliff failure mode as "primacy-biased document
# continuation overflow" — the chat-template anchor at the start of the
# prompt gets overpowered by the haystack continuation prior. This script
# tests two cheap interventions that strengthen the anchor without
# touching the model:
#
#   ARM 1: BASELINE
#     Just haystack + final question. The current niah_test.sh format.
#
#   ARM 2: PQRI (Periodic Question Re-Injection)
#     Insert "[REMINDER: <question>]" markers inside the haystack every
#     ~256 tokens. The chat template anchor is conceptually refreshed
#     because the question reappears throughout, not just at the end.
#
#   ARM 3: CONVCHUNK (Conversational Chunking)
#     Split the haystack into N=4 chunks, wrap each as a separate
#     <|user|> turn with the SAME question, terminated by <|eot_id|>.
#     The chat template anchor is *literally* refreshed at every turn
#     boundary because each chunk is a fresh user message.
#
# Grid: 3B Q4 × 4 contexts (1024, 1280, 1536, 2048) × 3 needles × 3 arms
#       = 36 trials. ~30 min on Metal.

set -e
export LC_ALL=C
export LANG=C

TQ=${TQ:-./build_metal/quant}
MODEL=${MODEL:-models/Llama-3.2-3B-Instruct-Q8_0.gguf}
THREADS=${THREADS:-8}
OUT_DIR=bench/results/niah
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
RAW_LOG="$OUT_DIR/raw_anchor_${RUN_ID}.log"
RESULT_CSV="$OUT_DIR/results_anchor_${RUN_ID}.csv"

mkdir -p "$OUT_DIR"
echo "arm,context,needle_idx,pass,response" > "$RESULT_CSV"

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

CONTEXTS=(1024 1280 1536 2048)
ARMS=("baseline" "pqri" "convchunk")
DEPTH=0.5

# ----------------------------------------------------------------------------
# Prompt builders — one per arm
# ----------------------------------------------------------------------------

build_prompt_baseline() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
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

build_prompt_pqri() {
  # Same haystack + needle layout as baseline, but with periodic reminder
  # markers inserted every ~256 tokens (~920 chars).
  python3 - "$1" "$2" "$3" <<'PYEOF'
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

# Reminder format — minimal, recognisable, not chat-template-token-disruptive
reminder = f" [REMINDER: {question}] "

# Insert at every ~920 chars (~256 tokens) at sentence boundaries
INTERVAL = 920
parts = []
pos = 0
while pos < len(h):
  end_pos = min(pos + INTERVAL, len(h))
  # Snap to next sentence boundary
  if end_pos < len(h):
    sb_next = h.find(". ", end_pos)
    if sb_next > 0 and sb_next - end_pos < 200:
      end_pos = sb_next + 2
  parts.append(h[pos:end_pos])
  pos = end_pos

augmented = reminder.join(parts)
sys.stdout.write(augmented + "\n\nQuestion: " + question)
PYEOF
}

build_prompt_convchunk() {
  # Split the haystack into 4 user turns, each terminated by <|eot_id|>,
  # with the question repeated at every turn. The chat template anchor
  # is *literally* refreshed because each chunk is a fresh user message
  # with its own header.
  python3 - "$1" "$2" "$3" <<'PYEOF'
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

# Split into 4 roughly equal chunks at sentence boundaries
N_CHUNKS = 4
chunks = []
chunk_target = len(h) // N_CHUNKS
pos = 0
for i in range(N_CHUNKS - 1):
  end_pos = pos + chunk_target
  sb_next = h.find(". ", end_pos)
  if sb_next > 0 and sb_next - end_pos < 200:
    end_pos = sb_next + 2
  chunks.append(h[pos:end_pos])
  pos = end_pos
chunks.append(h[pos:])

# Build the chat-format prompt manually (we will NOT use --chat for this arm)
# because we need raw control over the multi-turn structure.
prompt = ""
for i, ch in enumerate(chunks):
  prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
  prompt += f"Document part {i+1}/{N_CHUNKS}:\n{ch}\n\n"
  if i < N_CHUNKS - 1:
    prompt += "Acknowledged, continue with the next part."
  else:
    prompt += f"Question: {question}"
  prompt += "<|eot_id|>"
prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
sys.stdout.write(prompt)
PYEOF
}

# ----------------------------------------------------------------------------
# Run loop
# ----------------------------------------------------------------------------
total=$(( ${#ARMS[@]} * ${#CONTEXTS[@]} * ${#NEEDLES[@]} ))
run_idx=0

echo "==> Anchor mitigation test"
echo "    binary:  $TQ"
echo "    model:   $MODEL"
echo "    contexts: ${CONTEXTS[*]}"
echo "    arms:    ${ARMS[*]}"
echo "    needles: ${#NEEDLES[@]}  total: $total"
echo "    raw:     $RAW_LOG"
echo "    csv:     $RESULT_CSV"
echo ""

for arm in "${ARMS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    # Generous cli_ctx: wikitext (with @-@ and == markup) tokenises closer
    # to 3 chars/token than 4, and PQRI/convchunk add 100-300 tokens of
    # reminder/wrap overhead. Use 2× ctx + 256 to be safe everywhere.
    cli_ctx=$(( ctx * 2 + 256 ))
    for ni in "${!NEEDLES[@]}"; do
      run_idx=$(( run_idx + 1 ))
      needle="${NEEDLES[$ni]}"
      question="${QUESTIONS[$ni]}"
      keyword="${KEYWORDS[$ni]}"

      case "$arm" in
        baseline)
          prompt=$(build_prompt_baseline "$ctx" "$needle" "$question")
          chat_flag="--chat"
          ;;
        pqri)
          prompt=$(build_prompt_pqri "$ctx" "$needle" "$question")
          chat_flag="--chat"
          ;;
        convchunk)
          prompt=$(build_prompt_convchunk "$ctx" "$needle" "$question")
          chat_flag=""  # prompt is already chat-formatted
          ;;
      esac

      printf "[%2d/%d] %-10s ctx=%-5d needle=%d  " \
        "$run_idx" "$total" "$arm" "$ctx" "$ni"

      out=$( "$TQ" "$MODEL" -p "$prompt" -n 32 -T 0.0 -j "$THREADS" \
              $chat_flag --ctx "$cli_ctx" -k fp32 2>&1 || true )

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
        pass=0; echo "FAIL: ${resp:0:55}"
      fi

      echo "$arm,$ctx,$ni,$pass,\"$resp_csv\"" >> "$RESULT_CSV"
      echo "===== $arm ctx=$ctx needle=$ni =====" >> "$RAW_LOG"
      echo "$out" >> "$RAW_LOG"
      echo "" >> "$RAW_LOG"
    done
  done
done

echo ""
echo "==> Summary by (arm × ctx):"
printf "  %-10s" "arm"
for ctx in "${CONTEXTS[@]}"; do printf " %7d" "$ctx"; done
echo ""
for arm in "${ARMS[@]}"; do
  printf "  %-10s" "$arm"
  for ctx in "${CONTEXTS[@]}"; do
    pass=$(awk -F, -v a="$arm" -v c="$ctx" 'NR>1 && $1==a && $2==c {p+=$4; t++} END{if(t>0)printf "%d/%d", p, t; else print "n/a"}' "$RESULT_CSV")
    printf " %7s" "$pass"
  done
  echo ""
done
