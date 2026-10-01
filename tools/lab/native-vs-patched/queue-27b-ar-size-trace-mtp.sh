#!/bin/bash
# Build the adaptive-size-trace experiment and record a per-call AllReduce size/provider histogram
# under MTP depth 5 (draft + verify ARs) (adaptive pp1024 -3.9% investigation, PGC09).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export TRACE_EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 5"
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-size-trace adaptive-size-trace
SCRIPT ar-size-trace-mtp-1 $L/ar-size-trace.sh @b-27b-size-trace /mnt/data/bigcherry-work/runs/ar-size-trace-mtp-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
