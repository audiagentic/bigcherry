#!/bin/bash
# GP11: order-balanced N-arm end-to-end comparison on a real llama-server.
#
# Replaces replay-bench.sh, replay-bench-balanced.sh and isolate-winners.sh,
# which were three near-copies differing only in their arm list. Every result
# recorded in README.md is reproducible from this one script by setting ARMS;
# the exact ARMS string for each is written next to that result.
#
# WHY BALANCED: the first run of this comparison gave each arm one sample in a
# fixed order with the arm of interest last, and it lost on all six metrics.
# That is also exactly what thermal drift over a ~10 minute run produces, so
# arm and position were perfectly confounded and the result meant nothing.
# Here the arm order rotates every round, so over any multiple of (number of
# arms) rounds each arm occupies each position equally and monotone drift
# cancels. Position is recorded per row so drift can be TESTED, not assumed
# away. Measured position spread on this host is 0.19-0.35% -- the same order
# as the effects being chased, which is why this is not optional.
#
# ARMS: space-separated NAME:BUILD_DIGEST:CACHE triples, CACHE = yes|no.
#   yes  -> GGML_HIP_DISPATCH_CACHE points at $CACHE_PATH (winners active)
#   no   -> the variable is unset (no winners)
# Holding the digest constant across two arms and varying only CACHE isolates
# the winners from the build variant -- which is how the replay regression was
# finally attributed.
set -u
H=/mnt/vault/development/llmhosts/llamacpp
M=${MODEL:-/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf}
CACHE_PATH=${CACHE_PATH:-$HOME/.cache/bigcherry/tune-campaigns/b4d3cf708425/dispatch.cache}
BENCH_MODEL=${BENCH_MODEL:-qwen27b}
DEVICES=${DEVICES:-0,1}
PORT=${PORT:-18400}
ROUNDS=${ROUNDS:-6}
OUT=${OUT:-$HOME/ab-balanced.log}
ARMS=${ARMS:?set ARMS to space-separated NAME:DIGEST:CACHE triples}

read -r -a ARM_LIST <<< "$ARMS"
NARMS=${#ARM_LIST[@]}

resolve () {  # digest -> binary path
  find ~/.cache/bigcherry/builds -path "*/$1/bin/llama-server" | head -1
}
for spec in "${ARM_LIST[@]}"; do
  IFS=: read -r name digest usecache <<< "$spec"
  b=$(resolve "$digest")
  [ -x "$b" ] || { echo "arm $name: no binary for digest $digest" >&2; exit 1; }
  [ "$usecache" = "yes" ] && { [ -f "$CACHE_PATH" ] || { echo "arm $name wants cache but $CACHE_PATH missing" >&2; exit 1; }; }
done

cell () {
  local round=$1 pos=$2 name=$3 digest=$4 usecache=$5
  local bin; bin=$(resolve "$digest")
  export ROCR_VISIBLE_DEVICES=$DEVICES HIP_VISIBLE_DEVICES=$DEVICES LD_LIBRARY_PATH=$(dirname "$bin")
  if [ "$usecache" = "yes" ]; then export GGML_HIP_DISPATCH_CACHE=$CACHE_PATH; else unset GGML_HIP_DISPATCH_CACHE; fi
  rm -f /tmp/r.log
  # Patch 0800_server_shutdown_endpoint only REGISTERS the /shutdown route when
  # this is set -- without it the POST 404s and teardown falls back to kill -9,
  # which is what silently destroyed the replay hit/miss report on every run
  # before this. Setting the variable is not optional decoration; it is what
  # makes graceful shutdown exist at all.
  export LLAMA_SERVER_ENABLE_SHUTDOWN=1
  nohup "$bin" -m "$M" --port $PORT --host 127.0.0.1 --parallel 1 --metrics \
    -sm tensor --fit off --spec-type draft-mtp --spec-draft-n-max 4 > /tmp/r.log 2>&1 &
  local P=$!
  local ok=0; for i in $(seq 1 150); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && { ok=1; break; }; sleep 3; done
  if [ $ok -eq 0 ]; then
    echo "round=$round pos=$pos arm=$name SERVER_FAIL $(tail -2 /tmp/r.log|tr '\n' ' ')" >> "$OUT"
    kill -9 $P 2>/dev/null; sleep 5; return
  fi
  # The documented server-bench harness -- llama-bench cannot see MTP.
  local R=$(cd $H && timeout 3600 python3 bench/run_bench.py --bench-type server-bench \
      --server-url http://127.0.0.1:$PORT --model $BENCH_MODEL --bench-configs mtp-dual 2>&1 \
      | grep -aE "_tps:" | tr -s " " | tr "\n" " ")
  echo "round=$round pos=$pos arm=$name ${R:-NO_RESULT}" >> "$OUT"
  # MTP draft acceptance: two arms accepting drafts at different rates are not
  # doing the same work, and comparing their throughput is meaningless however
  # clean the numbers look.
  local ACC=$(grep -aoE "draft acceptance = [0-9.]+" /tmp/r.log | tail -1)
  echo "round=$round pos=$pos arm=$name mtp: ${ACC:-none}" >> "$OUT"
  # Did the cache actually LOAD? Logged at startup, so it survives any teardown.
  local LOADED=$(grep -a "replay cache .* loaded" /tmp/r.log | tail -1)
  echo "round=$round pos=$pos arm=$name cacheload: ${LOADED:-none}" >> "$OUT"
  # Shut down through /shutdown, NOT kill -9.
  #
  # An earlier version of this harness used kill -9 on the grounds that no arm
  # records tune measurements, so nothing buffered would be lost. That reasoning
  # was incomplete: the replay hit/miss report ("replay v2 N winner(s);
  # exact=.. miss=..") is emitted by the coverage reporter at SHUTDOWN, and
  # kill -9 destroys it. So every run to date measured a cache arm without ever
  # confirming the cache was consulted, let alone hit -- which makes "the
  # winners are neutral" and "the cache never resolved" indistinguishable.
  # That is the difference between a result and no result.
  if ! curl -sf -X POST -o /dev/null "http://127.0.0.1:$PORT/shutdown" 2>/dev/null; then
    # Loud, not silent: a 404 here means LLAMA_SERVER_ENABLE_SHUTDOWN did not
    # reach the server or 0800 is not in this build, and the run is about to
    # lose exactly the evidence it exists to collect.
    echo "round=$round pos=$pos arm=$name WARN shutdown-endpoint-unavailable" >> "$OUT"
  fi
  for i in $(seq 1 30); do kill -0 $P 2>/dev/null || break; sleep 1; done
  kill -9 $P 2>/dev/null
  local REPLAYSTAT=$(grep -aE "replay v2 .* winner|dispatch coverage" /tmp/r.log | tail -2 | tr '\n' ' ')
  echo "round=$round pos=$pos arm=$name replaystat: ${REPLAYSTAT:-none}" >> "$OUT"
  sleep 6
}

: > "$OUT"
echo "start $(date -Is) rounds=$ROUNDS arms=$ARMS model=$M devices=$DEVICES" >> "$OUT"
for r in $(seq 1 $ROUNDS); do
  shift_by=$(( (r - 1) % NARMS ))
  pos=1
  for i in $(seq 0 $((NARMS - 1))); do
    spec=${ARM_LIST[$(( (i + shift_by) % NARMS ))]}
    IFS=: read -r name digest usecache <<< "$spec"
    cell $r $pos "$name" "$digest" "$usecache"
    pos=$((pos + 1))
  done
  echo "round $r done $(date -Is)" >> "$OUT"
done
echo AB_BALANCED_DONE >> "$OUT"
