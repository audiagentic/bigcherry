#!/bin/bash
# DeepSeek-V4-Flash load/layout probe on all four GPUs. Stops radiance-vllm (R9700) for the run and restarts it.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/deepseek-v4-flash/gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT dsv4-flash-probe-1 tools/lab/deepseek/probe-v4-flash.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/dsv4-flash-probe-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
