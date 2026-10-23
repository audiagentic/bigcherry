#!/bin/bash
# Queue the hostcall-free RCCL build + verification under the host lock (VIS=0,1,2,3: the tests use all cards).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=${BC_TMP_ROOT}/bc-rocm BC_MODEL=${BC_MODEL_ROOT}/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT rccl-nohostcall-1 tools/lab/rccl/build-rccl-nohostcall.sh ${BC_WORK_ROOT}/runs/rccl-nohostcall-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
