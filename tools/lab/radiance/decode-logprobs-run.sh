#!/bin/bash
# RAD15: decode-path quality of one radiance setting (decode_logprobs.py). Serves the model on the R9700 once with SETTING=0
# and once with SETTING=1 (same binary, home and flags), captures greedy top-20 logprobs from each and compares them.
#   RAD_BIN, RAD_HOME, MODEL, SETTING (e.g. BC_GFX12_MOE_GEMV); EXTRA: further server flags; ROCR (default 2, the R9700).
set -u
W=/mnt/vault/development/projects/bigcherry/workspaces/main/work
OUT=$W/decode-lp-$(date +%m%d-%H%M%S); mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=$((36000 + RANDOM % 2000))
for v in 0 1; do
  env ROCR_VISIBLE_DEVICES=${ROCR:-2} HIP_VISIBLE_DEVICES=0 RADIANCE_HOME="$RAD_HOME" "$SETTING=$v" \
    "$RAD_BIN" --model "$MODEL" --radiance-home "$RAD_HOME" --port $PORT --tp 1 --max-num-seqs 1 ${EXTRA:-} \
    > "$OUT/server-$v.log" 2>&1 &
  PID=$!
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  python3 "$HERE/decode_logprobs.py" capture http://127.0.0.1:$PORT "$OUT/arm-$v.json" --max-tokens ${MAX_TOKENS:-256}
  kill -INT $PID; wait $PID
done
grep -c "$SETTING on" "$OUT"/server-*.log
python3 "$HERE/decode_logprobs.py" compare "$OUT/arm-0.json" "$OUT/arm-1.json" | tee "$OUT/compare.txt"
echo "OUT=$OUT"
