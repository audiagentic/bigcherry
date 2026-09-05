#!/bin/bash
# GP11: replay vs control vs native, ORDER-BALANCED.
#
# Supersedes the single-sample replay-bench.sh for any claim about replay.
# That script ran each arm exactly once, in a fixed order, with replay last --
# so "replay lost on all six metrics" is indistinguishable from "the arm that
# runs third loses on all six metrics", which is exactly what thermal drift
# over a ~10 minute run would produce. The confound is fatal to the claim, not
# merely a widening of the error bars.
#
# Here each round runs all three arms in a ROTATED order, so across 3 rounds
# every arm occupies every position exactly once and any monotone drift with
# position cancels. ROUNDS should therefore be a multiple of 3.
set -u
H=/mnt/vault/development/llmhosts/llamacpp
M=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
CACHE=~/.cache/bigcherry/tune-campaigns/b4d3cf708425/dispatch.cache
PORT=18400
ROUNDS=${ROUNDS:-6}
OUT=${OUT:-$HOME/replay-balanced.log}

NATIVE=$(find ~/.cache/bigcherry/builds -path "*/a55fa53d6c9c63e01115aa09847f77eb/bin/llama-server" | head -1)
CONTROL=$(find ~/.cache/bigcherry/builds -path "*/df75a6d33c4d2d5342e567ca2a6b01ba/bin/llama-server" | head -1)
REPLAY=$(find ~/.cache/bigcherry/builds -path "*/e59994bc49764809b1b4b957d71e934d/bin/llama-server" | head -1)
for b in "$NATIVE" "$CONTROL" "$REPLAY"; do
  [ -x "$b" ] || { echo "missing binary: $b" >&2; exit 1; }
done

cell () {
  local round=$1 pos=$2 name=$3 bin=$4 usecache=$5
  export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$(dirname $bin)
  if [ "$usecache" = "yes" ]; then export GGML_HIP_DISPATCH_CACHE=$CACHE; else unset GGML_HIP_DISPATCH_CACHE; fi
  rm -f /tmp/r.log
  nohup "$bin" -m "$M" --port $PORT --host 127.0.0.1 --parallel 1 --metrics \
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
  # Position is recorded on every row so drift-with-position can be tested
  # directly rather than assumed away by the design.
  echo "round=$round pos=$pos arm=$name ${R:-NO_RESULT}" >> "$OUT"
  kill -9 $P 2>/dev/null; sleep 6
}

: > "$OUT"
echo "start $(date -Is) rounds=$ROUNDS" >> "$OUT"
for r in $(seq 1 $ROUNDS); do
  case $(( (r - 1) % 3 )) in
    0) ORDER="native control replay" ;;
    1) ORDER="control replay native" ;;
    2) ORDER="replay native control" ;;
  esac
  pos=1
  for arm in $ORDER; do
    case $arm in
      native)  cell $r $pos native  "$NATIVE"  no  ;;
      control) cell $r $pos control "$CONTROL" no  ;;
      replay)  cell $r $pos replay  "$REPLAY"  yes ;;
    esac
    pos=$((pos + 1))
  done
  echo "round $r done $(date -Is)" >> "$OUT"
done
echo REPLAY_BALANCED_DONE >> "$OUT"
