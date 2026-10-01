#!/bin/bash
# 3-GPU (2x 7900 XTX gfx1100 + R9700 gfx1201) 27B AllReduce matrix through the queue.
# Multi-arch builds (gfx1100,gfx1201); arms: RCCL, 1244 root3 internal pipeline, none.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2 BUILD b-3g-control - gfx1100,gfx1201
VIS=0,1,2 BUILD b-3g-root3 allreduce-root3 gfx1100,gfx1201
VIS=0,1,2 AB ab-3g-allreduce $L/server-ab-allreduce-3gpu.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
