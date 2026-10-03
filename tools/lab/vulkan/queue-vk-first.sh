#!/bin/bash
# BRVP03/RRVP05: build the stock Vulkan lane and run the first 27B Vulkan screening (RADV), then the Linux
# test suite. Stops radiance-vllm (R9700) for the run and restarts it.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-vk-stock vulkan-stock:vulkan-stock:vulkan-linux - -
VIS=0,1,2,3 SCRIPT vk-27b-screen tools/lab/vulkan/probe-27b-vk.sh @b-vk-stock /mnt/data/bigcherry-work/runs/vk-27b-screen/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
