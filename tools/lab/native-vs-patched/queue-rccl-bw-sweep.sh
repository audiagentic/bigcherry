#!/bin/bash
# Queue the RCCL prefill-size bandwidth sweep (dual XTX) under the host + GPU locks.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
SCRIPT rccl-bw-sweep-1 tools/lab/native-vs-patched/rccl-bw-sweep.sh /mnt/data/bigcherry-work/runs/rccl-bw-sweep-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
