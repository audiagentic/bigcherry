#!/bin/bash
# Build 1291+1292+1294 and run the 1294 A/B against the 1291+1292 build.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=${BC_TMP_ROOT}/bc-rocm
export BC_MODEL=${BC_MODEL_ROOT}/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
base=${BC_DEV_ROOT}/projects/bigcherry/workspaces/main/work/builds/5b196af232266bae5856251b03acfe28/14bcf2379b53a5114df9f4928a18d202/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-topk-1 bigcherry:stock:linux-multi ar-cpu-root-kpool-topk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-topk-ab-1 tools/lab/flash-next/topk-ab.sh $base @b-flash-topk-1 ${BC_WORK_ROOT}/runs/flashnext-topk-ab-1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
