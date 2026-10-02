#!/bin/bash
# Build the adaptive-size-trace experiment and record a per-call AllReduce size/provider histogram
# for a 1000-token prompt and a 64-token decode (adaptive pp1024 -3.9% investigation, PGC09).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-size-trace bigcherry:stock:linux-multi adaptive-size-trace gfx1100
SCRIPT ar-size-trace-1 $L/ar-size-trace.sh @b-27b-size-trace /mnt/data/bigcherry-work/runs/ar-size-trace-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
