#!/bin/bash
# Adaptive vs RCCL on plain decode (no MTP), 27B Q8_0 dual XTX: does the adaptive gain hold
# without MTP? Decides whether a llama-bench (no MTP) contract lane can carry the promotion.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl adaptive-wire-latency
AB ab-27b-adaptive-plain $L/server-ab-adaptive-plain.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
