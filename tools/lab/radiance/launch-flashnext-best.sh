#!/bin/bash
# Launch Qwen3.8-Flash-Next on Brutus in the current best layout, per
# docs/reference/engines/radiance/models/qwen3.8-flash-next.md ("Current best layout and numbers") and the harness config
# tools/lab/radiance/helper-ab.json, which this mirrors outside the harness.
#
# Layout: engine on the R9700 (--tp 1, libr4d,libr3moe,libref); one r3_moe_worker helper per XTX holding full-width
# experts (22 GiB each, hottest experts by route heat), helper-owned experts skipped on the R9700; n-gram table in host
# RAM; fp8 KV; MTP depth 3; 4096-token chunks; up to 6 sequences; 32K context.
# Helper kernels on by default (R3_MOE_FP16, R3_MOE_AHALF, R3_MOE_NATIVE, R3_MOE_WORKER_VECIO, FUSESTAGE, FUSEROUTE) are
# not repeated here.
#
# Needs a radiance build of production source with the helper hand-off:
#   bigcherry build --lane radiance:stock:linux-radiance --experiment moe-remote-handoff
#
# Usage:
#   bash tools/lab/radiance/launch-flashnext-best.sh [start] [--build DIR] [--port N] [--heat FILE]
#   bash tools/lab/radiance/launch-flashnext-best.sh stop
# Without --build, the newest build under $BIGCHERRY_WORK_ROOT/builds that has the helper binary is used.
set -u

WORK=${BIGCHERRY_WORK_ROOT:-/mnt/data/bigcherry-work}
RUN=$WORK/flashnext
PIDS=$RUN/pids
SHM=/r3-moe-helper
PORT=36080
HEAT=$WORK/runs/perf51/heat.rank0.hot-first
BUILD=""
CMD=start
[ $# -gt 0 ] && case "$1" in start|stop) CMD=$1; shift ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --heat) HEAT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

stop_all() {
  # engine first (SIGINT is its clean shutdown), then the helpers; each gets 60 s before SIGKILL
  for name in engine helper1 helper0; do
    [ -f "$PIDS/$name" ] || continue
    pid=$(cat "$PIDS/$name")
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid"
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && { echo "$name ($pid) did not stop; killing" >&2; kill -KILL "$pid"; }
    fi
    rm -f "$PIDS/$name"
  done
}

if [ "$CMD" = stop ]; then stop_all; echo "stopped"; exit 0; fi

for name in engine helper0 helper1; do
  if [ -f "$PIDS/$name" ] && kill -0 "$(cat "$PIDS/$name")" 2>/dev/null; then
    echo "FLASHNEXT_EXIT=98 already running ($name pid $(cat "$PIDS/$name")); run '$0 stop' first"; exit 98
  fi
done

if [ -z "$BUILD" ]; then
  for d in $(ls -td "$WORK"/builds/*/*/ 2>/dev/null); do
    [ -x "${d}radiance_home/helpers/r3_moe_worker" ] && [ -x "${d}bin/radiance" ] && { BUILD=${d%/}; break; }
  done
fi
RAD_BIN=$BUILD/bin/radiance
RAD_HOME=$BUILD/radiance_home
HELPER=$RAD_HOME/helpers/r3_moe_worker
[ -x "$RAD_BIN" ] && [ -x "$HELPER" ] || { echo "FLASHNEXT_EXIT=99 no build with bin/radiance and the r3_moe_worker helper (--build '$BUILD')"; exit 99; }

MODEL=${MODEL:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
[ -f "$MODEL" ] || { echo "FLASHNEXT_EXIT=99 model not found: $MODEL"; exit 99; }
[ -f "$HEAT" ] || { echo "FLASHNEXT_EXIT=99 route-heat file not found: $HEAT (hottest-first placement is part of the best layout)"; exit 99; }

R9700=${R9700_ROCR:-2}
XTX=(${XTX_ROCR:-0 1})

# shared by the engine and both helpers (helper-ab.json "environment" + arm settings)
COMMON=(
  RADIANCE_HOME="$RAD_HOME"
  R3_MOE_WORKER_SHM="$SHM"
  R3_MOE_WORKER_COUNT=${#XTX[@]}
  R3_MOE_WORKER_ENGINE_RANKS=1
  R3_MOE_WORKER_MIB=22528
  R3_MOE_WORKER_MAX_ROWS=4096
  R3_MOE_WORKER_SMALL_ROWS=8
  R3_MOE_WORKER_MOE_DOWN_TILE=1
  R3_MOE_WORKER_HEAT="$HEAT"
  R3_MOE_WORKER_SKIP_OWNED=1
)

mkdir -p "$PIDS"
STAMP=$(date +%m%d-%H%M%S)
rm -f "/dev/shm${SHM}"
trap 'echo "start failed; stopping what was started" >&2; stop_all; exit 1' ERR

# helpers in order: helper 0 creates the segment, helper 1 attaches; each must report ready before the next starts
for i in "${!XTX[@]}"; do
  log=$RUN/helper$i-$STAMP.log
  env "${COMMON[@]}" ROCR_VISIBLE_DEVICES="${XTX[$i]}" HIP_VISIBLE_DEVICES=0 R3_MOE_WORKER_INDEX=$i \
    setsid "$HELPER" "$MODEL" > "$log" 2>&1 &
  echo $! > "$PIDS/helper$i"
  echo "helper$i on XTX ${XTX[$i]} (pid $!, log $log): waiting for ready..." >&2
  ready=0
  for _ in $(seq 900); do
    grep -qE '^r3_moe_worker: (ready |waiting for engine runtime expert maps)' "$log" && { ready=1; break; }
    kill -0 "$(cat "$PIDS/helper$i")" 2>/dev/null || break
    sleep 1
  done
  [ $ready = 1 ] || { echo "FLASHNEXT_EXIT=1 helper$i not ready; see $log"; tail -n 20 "$log"; false; }
done

LOG=$RUN/engine-$STAMP.log
env "${COMMON[@]}" R3_MOE_WORKER=1 ROCR_VISIBLE_DEVICES="$R9700" HIP_VISIBLE_DEVICES=0 \
  setsid "$RAD_BIN" --model "$MODEL" --radiance-home "$RAD_HOME" --port "$PORT" \
    --kernels libr4d,libr3moe,libref \
    --placement expert_tiered \
    --ngram-placement ram \
    --host-pool-mib 22528 \
    --tp 1 \
    --max-num-seqs 6 \
    --max-num-batched-tokens 4096 \
    --checkpoint-interval 4096 \
    --kv-cache-dtype fp8 \
    --num-speculative-tokens 3 \
    --max-model-len 32768 \
  > "$LOG" 2>&1 &
echo $! > "$PIDS/engine"
echo "engine on R9700 (pid $!, log $LOG): waiting for /health..." >&2
up=0
for _ in $(seq 180); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$(cat "$PIDS/engine")" 2>/dev/null || break
  sleep 5
done
[ $up = 1 ] || { echo "FLASHNEXT_EXIT=1 engine did not come up; see $LOG"; tail -n 30 "$LOG"; false; }
trap - ERR

cat <<INFO
FLASHNEXT_EXIT=0
Flash-Next is up: http://127.0.0.1:$PORT (OpenAI-compatible /v1/chat/completions, /v1/completions)
  build    $BUILD
  engine   R9700 pid $(cat "$PIDS/engine"), log $LOG
  helpers  XTX ${XTX[*]}, pids $(cat "$PIDS/helper0") $(cat "$PIDS/helper1" 2>/dev/null), logs $RUN/helper*-$STAMP.log
  stop     bash $0 stop
INFO
