#!/bin/bash
# GP11: is the replay regression caused by the WINNERS, or by the replay BUILD?
#
# replay-bench-balanced.sh established that the replay arm is slower than
# control with complete separation on five metrics. But those two arms are
# different binaries -- replay is built with GGML_HIP_DISPATCH_REPLAY=ON --
# so "the 19 winners are harmful" and "the replay build variant is slower"
# both fit that result. Attributing it to the winners without this run would
# be exactly the kind of unearned causal claim the balanced rerun was meant
# to stop.
#
# Here BOTH arms are the SAME replay binary. The only difference is whether
# GGML_HIP_DISPATCH_CACHE is set, i.e. whether the 19 winners are active.
# Build variant is held constant by construction, so any difference is the
# winners and nothing else.
#
# Arm order alternates every round for the same reason as the balanced run.
set -u
H=/mnt/vault/development/llmhosts/llamacpp
M=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
CACHE=~/.cache/bigcherry/tune-campaigns/b4d3cf708425/dispatch.cache
PORT=18400
ROUNDS=${ROUNDS:-8}
OUT=${OUT:-$HOME/isolate-winners.log}

REPLAY=$(find ~/.cache/bigcherry/builds -path "*/e59994bc49764809b1b4b957d71e934d/bin/llama-server" | head -1)
[ -x "$REPLAY" ] || { echo "missing replay binary" >&2; exit 1; }
[ -f "$CACHE" ] || { echo "missing winner cache: $CACHE" >&2; exit 1; }

cell () {
  local round=$1 pos=$2 name=$3 usecache=$4
  export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$(dirname $REPLAY)
  if [ "$usecache" = "yes" ]; then export GGML_HIP_DISPATCH_CACHE=$CACHE; else unset GGML_HIP_DISPATCH_CACHE; fi
  rm -f /tmp/r.log
  nohup "$REPLAY" -m "$M" --port $PORT --host 127.0.0.1 --parallel 1 --metrics \
    -sm tensor --fit off --spec-type draft-mtp --spec-draft-n-max 4 > /tmp/r.log 2>&1 &
  local P=$!
  local ok=0; for i in $(seq 1 150); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && { ok=1; break; }; sleep 3; done
  if [ $ok -eq 0 ]; then
    echo "round=$round pos=$pos arm=$name SERVER_FAIL $(tail -2 /tmp/r.log|tr '\n' ' ')" >> "$OUT"
    kill -9 $P 2>/dev/null; sleep 5; return
  fi
  local R=$(cd $H && timeout 3600 python3 bench/run_bench.py --bench-type server-bench \
      --server-url http://127.0.0.1:$PORT --model qwen27b --bench-configs mtp-dual 2>&1 \
      | grep -aE "_tps:" | tr -s " " | tr "\n" " ")
  echo "round=$round pos=$pos arm=$name ${R:-NO_RESULT}" >> "$OUT"
  # MTP draft acceptance: if the two arms accept drafts at different rates they
  # are not running the same amount of work, and a throughput comparison
  # between them is meaningless regardless of how clean the numbers look.
  local ACC=$(grep -aoE "(accept[^,]*|draft[^,]*)" /tmp/r.log | tail -3 | tr '\n' ' ')
  echo "round=$round pos=$pos arm=$name mtp: ${ACC:-none}" >> "$OUT"
  kill -9 $P 2>/dev/null; sleep 6
}

: > "$OUT"
echo "start $(date -Is) rounds=$ROUNDS binary=$REPLAY" >> "$OUT"
for r in $(seq 1 $ROUNDS); do
  if [ $(( r % 2 )) -eq 1 ]; then ORDER="nocache winners"; else ORDER="winners nocache"; fi
  pos=1
  for arm in $ORDER; do
    case $arm in
      nocache) cell $r $pos nocache no  ;;
      winners) cell $r $pos winners yes ;;
    esac
    pos=$((pos + 1))
  done
  echo "round $r done $(date -Is)" >> "$OUT"
done
echo ISOLATE_WINNERS_DONE >> "$OUT"
