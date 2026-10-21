#!/bin/bash
# Launch Qwen3.8-Flash-Next on Brutus with the current best known layout (R9700 engine + two XTX expert helpers),
# per docs/reference/engines/radiance/models/qwen3.8-flash-next.md ("Current best layout and numbers").
#
# Layout: engine on the R9700 (libr4d,libr3moe,libref), one r3_moe_worker per XTX (full-width experts, 22 GiB each,
# helper-owned experts skipped on the R9700, hottest experts routed to the helpers), n-gram table in host RAM,
# fp8 KV cache, MTP depth 3, 4096-token chunks, up to 6 concurrent sequences.
#
# Usage:
#   RAD_BIN=/path/to/radiance RAD_HOME=/path/to/radiance_home MODEL=/path/to/model.rad \
#     bash tools/lab/radiance/launch-flashnext-best.sh [--port N] [--heat /path/to/heat/file]
#
# If RAD_BIN/RAD_HOME are not set, the script looks for the most recently built radiance under
# /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/*/*/bin/radiance (Brutus layout).
# If MODEL is not set, it defaults to the Flash-Next container at its usual Brutus path.
# --heat points at a saved route-heat histogram (R3_MOE_WORKER_HEAT) for hottest-first expert placement; without
# one, the worker falls back to its own default placement (coldest-first measured ~4% slower prefill, RAD06).
set -u

WORK=/mnt/vault/development/projects/bigcherry/workspaces/main/work
PORT=36080
HEAT="${HEAT_DEFAULT:-/mnt/data/bigcherry-work/runs/perf51/heat.rank0.hot-first}"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --heat) HEAT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${RAD_BIN:-}" ] || [ -z "${RAD_HOME:-}" ]; then
  B=$(ls -td "$WORK"/builds/*/*/ 2>/dev/null | head -1)
  [ -n "$B" ] && [ -x "${B}bin/radiance" ] || { echo "FLASHNEXT_EXIT=99 no RAD_BIN/RAD_HOME given and none found under $WORK/builds"; exit 99; }
  RAD_BIN=${RAD_BIN:-${B}bin/radiance}
  RAD_HOME=${RAD_HOME:-${B}radiance_home}
fi
[ -x "$RAD_BIN" ] || { echo "FLASHNEXT_EXIT=99 RAD_BIN not executable: $RAD_BIN"; exit 99; }

MODEL=${MODEL:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
[ -f "$MODEL" ] || { echo "FLASHNEXT_EXIT=99 model not found: $MODEL"; exit 99; }

R9700_BDF=${R9700_BDF:-0000:09:00.0}
XTX0=${XTX0:-0}
XTX1=${XTX1:-1}

HEAT_ENV=()
if [ -n "$HEAT" ] && [ -f "$HEAT" ]; then
  HEAT_ENV=(R3_MOE_WORKER_HEAT="$HEAT")
else
  [ -n "$HEAT" ] && echo "note: heat file not found at $HEAT (stale perf51 run?); falling back to the worker's default placement." >&2
  HEAT=""
  echo "note: no usable --heat histogram; the worker falls back to its own default placement (measured ~4% slower prefill than hottest-first, RAD06)." >&2
fi

LOG=$WORK/flashnext-$(date +%m%d-%H%M%S).log

env \
  ROCR_VISIBLE_DEVICES=2 HIP_VISIBLE_DEVICES=0 \
  RADIANCE_HOME="$RAD_HOME" \
  R3_MOE_WORKER=1 \
  R3_MOE_WORKER_COUNT=2 \
  R3_MOE_WORKER_DEVICES="$XTX0 $XTX1" \
  R3_MOE_WORKER_ENGINE_RANKS=1 \
  R3_MOE_WORKER_MIB=22528 \
  R3_MOE_WORKER_SKIP_OWNED=1 \
  R3_MOE_WORKER_MAX_ROWS=4096 \
  R3_MOE_WORKER_SMALL_ROWS=8 \
  R3_MOE_FP16=1 \
  R3_MOE_AHALF=1 \
  R3_MOE_WORKER_VECIO=1 \
  "${HEAT_ENV[@]}" \
  "$RAD_BIN" --model "$MODEL" --radiance-home "$RAD_HOME" --port "$PORT" \
    --kernels libr4d,libr3moe,libref \
    --placement expert_tiered \
    --ngram-placement ram \
    --host-pool-mib 30720 \
    --tp 1 \
    --max-num-seqs 6 \
    --max-num-batched-tokens 4096 \
    --checkpoint-interval 4096 \
    --kv-cache-dtype fp8 \
    --num-speculative-tokens 3 \
  > "$LOG" 2>&1 &
PID=$!

echo "waiting for server to come up (PID $PID, log $LOG)..." >&2
UP=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then UP=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then break; fi
  sleep 5
done

if [ "$UP" != 1 ]; then
  echo "FLASHNEXT_EXIT=1 server did not come up; see $LOG"
  tail -n 30 "$LOG"
  exit 1
fi

cat <<INFO
FLASHNEXT_EXIT=0
----------------------------------------------------------------------
Flash-Next is up.
  PID:        $PID
  Log:        $LOG
  Model:      $MODEL
  Endpoint:   http://127.0.0.1:$PORT  (OpenAI-compatible: /v1/chat/completions, /v1/completions)
  Health:     http://127.0.0.1:$PORT/health
  Engine:     R9700 (ROCR_VISIBLE_DEVICES=2), kernels libr4d,libr3moe,libref
  Helpers:    2x r3_moe_worker on XTX devices $XTX0,$XTX1 (full-width experts, 22 GiB each, hottest-first$([ -z "$HEAT" ] && echo " -- NOT configured, see note above" || echo ", heat=$HEAT"))
  KV cache:   fp8, host pool 30720 MiB, n-gram table in RAM
  Speculative: MTP depth 3, max-num-seqs 6, chunk 4096
Stop with: kill -INT $PID
----------------------------------------------------------------------
INFO
