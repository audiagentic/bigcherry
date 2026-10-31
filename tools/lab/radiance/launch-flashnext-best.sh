#!/bin/bash
# Launch Qwen3.8-Flash-Next on dev in the current best layout, per
# docs/reference/engines/radiance/models/qwen3.8-flash-next.md ("Current best layout and numbers") and the harness config
# tools/lab/radiance/helper-ab.json, which this mirrors outside the harness.
#
# Offload: the engine card runs the model (--tp 1); one r3_moe_worker helper per helper card holds full-width experts (the hottest by route
# heat, up to the layout's budget), helper-owned experts skipped on the engine card; n-gram table in host
# RAM; fp8 KV; MTP depth 3; 2048-token chunks; up to 2 sequences; 1,048,576-token context (the 262,144 the model was trained
# on, stretched 4x by --rope-scaling yarn:4). A prefix hit resumes at the nearest cached token (--prefix-resume nearest) rather
# than on the cold run's chunk grid: re-sent prefixes cost 0.2-0.3 s a turn instead of 0.6-0.8 s, at the price of greedy text
# that can differ from a cold run's near ties. The cache takes the VRAM the experts leave, which costs no decode or
# 4K-prefill speed (RAD37 measurements, qwen3.8-flash-next.md); one sequence is 30% slower at decode, so keep two.
# Helper kernels on by default (R3_MOE_FP16, R3_MOE_AHALF, R3_MOE_NATIVE, R3_MOE_WORKER_VECIO, FUSESTAGE, FUSEROUTE) are
# not repeated here.
#
# Which cards play which part is a layout (engines/radiance/layouts.toml, `bigcherry layout list|show NAME`), resolved against this host's device
# inventory; card types (plugin, weight formats, few-rows kernel) are engines/radiance/archs.toml. --layout NAME (or FLASHNEXT_LAYOUT) picks one,
# default flashnext-2xtx; --budget ARCH=MIB overrides the VRAM budget of the group using that architecture (e.g. --layout flashnext-2xtx-rdna2-tail
# --budget gfx1030=4096). A layout that needs a card the host does not have, or a plugin the build lacks, stops here with the reason.
#
# Needs a radiance build of production source with the helper hand-off:
#   bigcherry build --lane radiance:stock:linux-radiance --experiment moe-remote-handoff
#
# Usage:
#   bash tools/lab/radiance/launch-flashnext-best.sh [start] [--build DIR] [--port N] [--heat FILE] [--host-pool-mib N] [--chunk TOKENS] [--ngram-placement ram|disk|vram] [--kv-dtype fp8|fp8_v4|k6_v4] [--engine-arg ARG ...] [--ngram-device INDEX] [--max-model-len TOKENS] [--record-heat PREFIX [--record-period S]] [--gpu-power] [--layout NAME] [--budget ARCH=MIB]
#   bash tools/lab/radiance/launch-flashnext-best.sh stop
# Without --build, the newest build under $BC_WORK_ROOT/builds that has the helper binary is used.
# --heat FILE: the route-heat file that ranks the experts, hottest first (uint32[128][512] = max - count of a recorded histogram; heat-placement/heat-invert.py
# makes one). --record-heat PREFIX: the engine counts every routed expert in VRAM and rewrites PREFIX.rank0 every --record-period seconds (default 30)
# while serving: a recording run (the counting costs speed; do not serve production with it). Invert the file, restart with --heat and a larger
# --budget (RHP10: workload-recorded heat plus gfx1100=23500 gave +34% prefill at 675K used context on a corpus it had not seen).
# --host-pool-mib N: pinned host memory the expert and KV tiers share (default 22528; more gives the cache more of a long context: the engine log
# reports how much of the worst case it backs).
# --chunk TOKENS: the largest step (default 2048): the activation arena and the prefill staging are sized from it, so a smaller chunk leaves the KV cache more VRAM.
# --ngram-placement ram|disk|vram: where the 47.7 GiB n-gram table lives (default ram: pinned host memory the cards read). disk reads its rows from the
# container and frees that RAM for --host-pool-mib.
# --kv-dtype T: the KV cache type (default fp8; fp8_v4 costs 0.78 of it per token and k6_v4 0.69, both lossy). --engine-arg ARG: one more engine argument,
# repeatable (an option and its value are two --engine-arg, e.g. --engine-arg --kv-resident --engine-arg 65536).
# --ngram-device INDEX: an inventory card (not the engine, not a helper) that holds as much of the n-gram table as fits and gathers its rows (libr3ngram, R3_NGRAM_DEV);
# the rows past its VRAM stay in a host copy. The card then sits in the engine process beside the engine card.
# --max-model-len TOKENS: the longest context a sequence may have (default 1048576).
# --rope-scaling S: default yarn:4 (the 1M stretch); use none for contexts within the trained 262144.
# --max-num-seqs N: sessions served at once (default 2; each costs per-sequence state in VRAM and checkpoint slots in pinned RAM).
# --prefix-resume MODE: default nearest; off leaves the flag out (engines before 1.14.0 do not have it).
# --gpu-power (or GPU_POWER=1): apply the cards' configured power settings ([host.*.devices.power]) for this run and
# put back what the cards ran before at stop. Without write access it lists the setup steps still missing and runs
# with the cards as they are.
# HELPER_WRAP=<command>: run each helper under it (e.g. a profiler wrapper; R3_MOE_WORKER_INDEX tells it which helper).
set -u

WORK=${BC_WORK_ROOT:?set BC_WORK_ROOT (source tools/env/bigcherry-env.sh)}
RUN=$WORK/flashnext
PIDS=$RUN/pids
SHM=/r3-moe-helper
PORT=36080
HEAT=$WORK/runs/perf51/heat.rank0.hot-first
BUILD=""
GPU_POWER=${GPU_POWER:-0}
LAYOUT=${FLASHNEXT_LAYOUT:-flashnext-2xtx}
BUDGETS=()
RECORD=""
HOST_POOL_MIB=22528
CHUNK=2048
NGRAM=ram
KV_DTYPE=fp8
ENGINE_ARGS=()
NGRAM_DEV=""
MAX_LEN=1048576
ROPE=yarn:4
SEQS=2
PREFIX_RESUME=nearest
RECORD_PERIOD=30
CMD=start
[ $# -gt 0 ] && case "$1" in start|stop) CMD=$1; shift ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --heat) HEAT=$2; shift 2 ;;
    --gpu-power) GPU_POWER=1; shift ;;
    --layout) LAYOUT=$2; shift 2 ;;
    --budget) BUDGETS+=("$2"); shift 2 ;;
    --host-pool-mib) HOST_POOL_MIB=$2; shift 2 ;;
    --chunk) CHUNK=$2; shift 2 ;;
    --kv-dtype) KV_DTYPE=$2; shift 2 ;;
    --engine-arg) ENGINE_ARGS+=("$2"); shift 2 ;;
    --ngram-device) NGRAM_DEV=$2; shift 2 ;;
    --max-model-len) MAX_LEN=$2; shift 2 ;;
    --rope-scaling) ROPE=$2; shift 2 ;;
    --max-num-seqs) SEQS=$2; shift 2 ;;
    --prefix-resume) PREFIX_RESUME=$2; shift 2 ;;
    --ngram-placement) NGRAM=$2; shift 2 ;;
    --record-heat) RECORD=$2; shift 2 ;;
    --record-period) RECORD_PERIOD=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

stop_all() {
  # engine first (SIGINT is its clean shutdown), then the helpers; each gets 60 s before SIGKILL
  for name in engine $(cd "$PIDS" 2>/dev/null && ls -r helper* 2>/dev/null); do
    [ -f "$PIDS/$name" ] || continue
    pid=$(cat "$PIDS/$name")
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid"
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && { echo "$name ($pid) did not stop; killing" >&2; kill -KILL "$pid"; }
    fi
    rm -f "$PIDS/$name"
  done
  # power settings this launcher applied go back to what the cards ran before, once the processes are gone
  if [ -f "$RUN/gpu-power" ]; then
    PYTHONPATH=tools python3 -m bigcherry gpu-power restore >&2 && rm -f "$RUN/gpu-power"
  fi
}

if [ "$CMD" = stop ]; then stop_all; echo "stopped"; exit 0; fi

for name in engine $(cd "$PIDS" 2>/dev/null && ls helper* 2>/dev/null); do
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

# Warn (never block) when the host lacks settings the tooling expects: configured per-card power settings not applied,
# CPU energy counters unreadable (`bigcherry host-check`, PWR04).
PYTHONPATH=tools python3 -m bigcherry host-check 2>/dev/null | grep '^WARN' >&2 || true

# Optional per-run power settings (PWR04); when they cannot be written, gpu-power lists the missing setup steps and the run goes
# ahead with the cards as they are.
if [ "$GPU_POWER" = 1 ]; then
  mkdir -p "$RUN"
  if PYTHONPATH=tools python3 -m bigcherry gpu-power apply >&2; then
    touch "$RUN/gpu-power"
  else
    echo "gpu-power: running with the cards as they are" >&2
  fi
fi

MODEL=${MODEL:-${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
[ -f "$MODEL" ] || { echo "FLASHNEXT_EXIT=99 model not found: $MODEL"; exit 99; }
[ -f "$HEAT" ] || { echo "FLASHNEXT_EXIT=99 route-heat file not found: $HEAT (hottest-first placement is part of the best layout)"; exit 99; }

source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
# Which card does what comes from the layout (engines/radiance/layouts.toml) resolved against this host's device inventory, and what a card
# type needs (plugin, weight formats, few-rows kernel) from engines/radiance/archs.toml: nothing about cards is named here.
BUDGET_ARGS=(); for b in "${BUDGETS[@]:-}"; do [ -n "$b" ] && BUDGET_ARGS+=(--budget "$b"); done
LAYOUT_OUT=$(PYTHONPATH=tools python3 -m bigcherry layout env "$LAYOUT" "${BUDGET_ARGS[@]}") || { echo "FLASHNEXT_EXIT=99 layout '$LAYOUT' does not resolve on this host (above)"; exit 99; }
eval "$LAYOUT_OUT"
R9700=$LAYOUT_ENGINE_DEVICE
HELPERS=(); for ((i = 0; i < LAYOUT_HELPER_COUNT; i++)); do HELPERS+=("$i"); done
for ((i = 0; i < LAYOUT_HELPER_COUNT; i++)); do
  declare -n h="LAYOUT_HELPER_$i"
  plugin=$(printf '%s\n' "${h[@]}" | sed -n 's/^R3_MOE_WORKER_PLUGIN_FILE=//p')
  [ -f "$RAD_HOME/kernels/$plugin" ] || { echo "FLASHNEXT_EXIT=99 layout '$LAYOUT' needs kernels/$plugin in the build ($RAD_HOME); rebuild with the radiance-lab lane"; exit 99; }
  unset -n h
done
echo "layout $LAYOUT: engine card $LAYOUT_ENGINE_DEVICE; $LAYOUT_SUMMARY" >&2

# shared by the engine and the helpers (helper-ab.json "environment" + arm settings), then what the layout says
COMMON=(
  RADIANCE_HOME="$RAD_HOME"
  R3_MOE_WORKER_SHM="$SHM"
  R3_MOE_WORKER_MAX_ROWS=4096
  R3_MOE_WORKER_SMALL_ROWS=8
  R3_MOE_WORKER_MOE_DOWN_TILE=1
  R3_MOE_WORKER_HEAT="$HEAT"
  R3_MOE_WORKER_SKIP_OWNED=1
  "${LAYOUT_COMMON[@]}"
)
if [ -n "$RECORD" ]; then
  COMMON+=(R3_MOE_WORKER_HISTOGRAM="$RECORD" R3_MOE_WORKER_HISTOGRAM_PERIOD_S="$RECORD_PERIOD")
  echo "recording the route histogram to $RECORD.rank0 every ${RECORD_PERIOD}s (costs speed; invert it with tools/lab/radiance/heat-placement/heat-invert.py for --heat)" >&2
fi

mkdir -p "$PIDS"
STAMP=$(date +%m%d-%H%M%S)
rm -f "/dev/shm${SHM}"
trap 'echo "start failed; stopping what was started" >&2; stop_all; exit 1' ERR

# helpers in order: helper 0 creates the segment, helper 1 attaches; each must report ready before the next starts
for i in "${!HELPERS[@]}"; do
  log=$RUN/helper$i-$STAMP.log
  declare -n hv="LAYOUT_HELPER_$i"
  env "${COMMON[@]}" "${hv[@]}" \
    setsid ${HELPER_WRAP:-} "$HELPER" "$MODEL" > "$log" 2>&1 &
  echo $! > "$PIDS/helper$i"
  unset -n hv
  echo "helper$i (pid $!, log $log): waiting for ready..." >&2
  ready=0
  for _ in $(seq 900); do
    grep -qE '^r3_moe_worker: (ready |waiting for engine runtime expert maps)' "$log" && { ready=1; break; }
    kill -0 "$(cat "$PIDS/helper$i")" 2>/dev/null || break
    sleep 1
  done
  [ $ready = 1 ] || { echo "FLASHNEXT_EXIT=1 helper$i not ready; see $log"; tail -n 20 "$log"; false; }
done

LOG=$RUN/engine-$STAMP.log
RESUME_ARGS=(); [ "$PREFIX_RESUME" = off ] || RESUME_ARGS=(--prefix-resume "$PREFIX_RESUME")
ENGINE_KERNELS="$LAYOUT_ENGINE_PLUGIN,libr3moe,libref"
ENGINE_VISIBLE="$R9700"
ENGINE_HIP=0
NGRAM_ENV=()
if [ -n "$NGRAM_DEV" ]; then
  # the n-gram card is visible to the engine process as HIP device 1 (the engine card is 0); the plugin goes ahead of the engine's library
  ENGINE_KERNELS="libr3ngram,$ENGINE_KERNELS"
  ENGINE_VISIBLE="$R9700,$NGRAM_DEV"
  ENGINE_HIP="0,1"
  NGRAM_ENV=(R3_NGRAM_DEV=1 R3_NGRAM_HELPER_SO="$RAD_HOME/helpers/libr3_ngram_helper.so" R3_NGRAM_AVX="$RAD_HOME/kernels/libavx.so")
fi
env "${COMMON[@]}" "${NGRAM_ENV[@]}" R3_MOE_WORKER=1 ROCR_VISIBLE_DEVICES="$ENGINE_VISIBLE" HIP_VISIBLE_DEVICES="$ENGINE_HIP" \
  setsid "$RAD_BIN" --model "$MODEL" --radiance-home "$RAD_HOME" --port "$PORT" \
    --kernels "$ENGINE_KERNELS" \
    --placement expert_tiered \
    --ngram-placement "$NGRAM" \
    --host-pool-mib "$HOST_POOL_MIB" \
    --tp 1 \
    --max-num-seqs "$SEQS" \
    --max-num-batched-tokens "$CHUNK" \
    --checkpoint-interval 4096 \
    --kv-cache-dtype "$KV_DTYPE" \
    --num-speculative-tokens 3 \
    --max-model-len "$MAX_LEN" \
    --rope-scaling "$ROPE" \
    "${RESUME_ARGS[@]}" \
    "${ENGINE_ARGS[@]}" \
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
  layout   $LAYOUT: $LAYOUT_SUMMARY\n  helpers  pids $(for p in "$PIDS"/helper*; do printf "%s " "$(cat "$p")"; done), logs $RUN/helper*-$STAMP.log
  stop     bash $0 stop
INFO
