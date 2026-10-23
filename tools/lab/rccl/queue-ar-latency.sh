#!/bin/bash
# Queue the RCCL small-message latency microbenchmark (stops/restarts radiance-vllm: the 3-rank case uses the R9700).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=${BC_TMP_ROOT}/bc-rocm
export BC_MODEL=${BC_MODEL_ROOT}/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT cpu-root-verify-2 tools/lab/rccl/cpu-root-ar.sh ${BC_WORK_ROOT}/runs/cpu-root-verify-2/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
