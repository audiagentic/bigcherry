#!/usr/bin/env bash
# RHP01 lease-holder only: instrument existing Radiance routing evidence, not the GPU route producer.
set -euo pipefail
if [[ $# != 2 || ! -d "$1" || -z "$2" ]]; then
  echo "usage: agent-lease acquire gpu -- bash tools/lab/radiance/heat-placement/rhp01-coverage.sh ABS_BUILD_DIR OUT_DIR" >&2
  exit 2
fi
BUILD="$1"
OUT="$2"
LAUNCH=tools/lab/radiance/launch-flashnext-best.sh
EXPERIMENT=tools/lab/radiance/heat-placement
mkdir -p "$OUT"

ACTIVE=0
start_server() {
  bash "$LAUNCH" start "$@"
  ACTIVE=1
}
stop_server() {
  if [[ "$ACTIVE" == 1 ]]; then
    bash "$LAUNCH" stop
    ACTIVE=0
  fi
}
# Never stop a server we did not start, including on a failed launch lock.
trap stop_server EXIT

run_stream() {
  local name="$1" streams="$2" prompt="$3" tokens="$4" reps="$5"
  PYTHONPATH=tools python3 -m bigcherry stream-bench \
    --url http://127.0.0.1:36080 --streams "$streams" \
    --prompt-tokens "$prompt" --max-tokens "$tokens" --reps "$reps" \
    --output "$OUT/$name.json"
}

for chunk in 2048 4096 8192; do
  for arm in off on; do
    echo "RHP01 arm=prefill chunk=$chunk evidence=$arm" >&2
    start_server --layout flashnext-4gpu --build "$BUILD" \
      --chunk "$chunk" --experiment "$EXPERIMENT/rhp01-evidence-$arm.json"
    run_stream "prefill-$chunk-$arm" 1 32768 64 3
    stop_server
    # Record the just-stopped engine/helper logs, manifest and source SHA per arm.
  done
done

start_server --layout flashnext-4gpu --build "$BUILD" \
  --chunk 2048 --experiment "$EXPERIMENT/rhp01-evidence-on.json"
run_stream "mtp-output-64" 1,4,8 2048 64 3
# At 2048 tokens per step, a 131072-token admitted prompt traverses the 32-bank ring.
# Actual accepted token count and real 64-row MTP verification require log evidence.
run_stream "ring-wrap-64-prefill-steps" 1 131072 64 1
stop_server
echo "RHP01 complete: archive per-arm engine RHP01 lines and run manifests before assessing coverage" >&2
