#!/bin/bash
# RAD32: performance of this checkout's radiance pin against the recorded v1.9.0 numbers, on the host's own cards.
#   27b   Qwen3.8-27B MXFP4, tensor parallel over the gfx1100 cards (libr3): decode and a 7.2K prefill, 3 repeats.
#         Baseline (docs/reference/engines/radiance/models/qwen3.8-27b.md, 2026-10-10/12): decode 46.7-47.2 tok/s,
#         prefill 2,251-2,256 tok/s.
#   fn    Flash-Next in the production layout (pwr01-baseline.sh): decode and prefill rates and the decode text
#         identity. Baseline: runs pwr01-baseline-10280942..10281145, decode 88.3-89.8 tok/s, prefill 3,265-3,345 tok/s.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad32-perf.sh [27b|fn|all=all] [repeats=3]
set -u
WHAT=${1:-all}; REPS=${2:-3}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-build.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad32-perf-$(date +%m%d%H%M); mkdir -p "$OUT"
git log -1 --format='commit %H %s' | tee "$OUT/summary.txt"
B=$(rad_build "$OUT/build.log") || { echo "BUILD_FAILED"; exit 1; }
echo "build $B" | tee -a "$OUT/summary.txt"

if [ "$WHAT" = 27b ] || [ "$WHAT" = all ]; then
    XTX=$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
print(",".join(str(d.index) for d in e.load_default().host(None).devices_for_arch("gfx1100")))')
    MODEL=${MODEL_27B:-${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
    [ -f "$MODEL" ] || { echo "27B model not found: $MODEL"; exit 1; }
    echo "== 27b on gfx1100 cards $XTX" | tee -a "$OUT/summary.txt"
    HIP_VISIBLE_DEVICES=$XTX ROCR_VISIBLE_DEVICES=$XTX RADIANCE_HOME=$B/radiance_home \
        PYTHONPATH=tools timeout 3000 python3 -m bigcherry engine-bench --engine radiance --binary "$B/bin/radiance" \
        --model "$MODEL" --out "$OUT/27b" --depth 7200 --decode 128 --reps "$REPS" --label rad32-27b --shutdown sigint \
        -- --radiance-home "$B/radiance_home" --tp "$(echo "$XTX" | tr ',' '\n' | wc -l)" --tp-wire wht6 \
           --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 1 > "$OUT/27b.log" 2>&1
    echo "27b rc=$?" | tee -a "$OUT/summary.txt"
    tail -12 "$OUT/27b.log" | tee -a "$OUT/summary.txt"
fi

if [ "$WHAT" = fn ] || [ "$WHAT" = all ]; then
    echo "== flash-next (pwr01-baseline.sh)" | tee -a "$OUT/summary.txt"
    for r in $(seq "$REPS"); do
        bash tools/lab/radiance/pwr01-baseline.sh 2>&1 | grep -E "^decode:|^prefill:|^decode text|^OUT" | tee -a "$OUT/summary.txt"
    done
fi
echo "OUT $OUT"
