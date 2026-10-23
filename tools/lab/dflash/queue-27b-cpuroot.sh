#!/bin/bash
# 1291 --allreduce cpu-root vs auto (adaptive on dual gfx1100) on Qwen3.8-27B Q8_0 dual XTX, plain and MTP5, ABBA.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=${BC_TMP_ROOT}/bc-rocm
export BC_MODEL=${BC_MODEL_ROOT}/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1 SCRIPT cpuroot-27b-1 tools/lab/dflash/probe-27b.sh @b-flash-cpuroot-4 ${BC_WORK_ROOT}/runs/cpuroot-27b-1/out (plain|cr27-.*) 3
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
