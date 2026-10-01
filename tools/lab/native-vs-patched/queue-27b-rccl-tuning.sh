#!/bin/bash
# RCCL tuning (no patch): dual XTX 27B Q8_0 MTP, 6 balanced rounds per group.
# Group 1 on the production binary: RCCL default vs NCCL_PROTO=LL vs LL128 (low-latency protocols
# target the small decode reductions where the host pipeline currently beats RCCL).
# Group 2 on the adaptive binary: RCCL+LL vs adaptive vs adaptive with RCCL+LL for its large side.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-control -
BUILD b-27b-adaptive allreduce-adaptive
AB ab-27b-rccl-proto $L/server-ab-rccl-proto.json --pairs 6
AB ab-27b-rccl-adaptive $L/server-ab-rccl-adaptive.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
