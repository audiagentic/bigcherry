#!/bin/bash
# Queue the dual-XTX RCCL env sweep (R9700/vLLM untouched).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1 SCRIPT rccl-env-sweep-1 tools/lab/rccl/rccl-env-sweep.sh @b-27b-awl /mnt/data/bigcherry-work/runs/rccl-env-sweep-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
