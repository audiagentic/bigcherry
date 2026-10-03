#!/bin/bash
# PRVP03 phase 0: build the Vulkan lane with 1290 and screen 27B -sm tensor with the provider on
# (BIGCHERRY_VK_ALLREDUCE=host-f32, activation marker traced) vs off (stock meta fallback) on the same
# binary, plus -sm layer as the no-collective reference. Stops/restarts radiance-vllm.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-vk-ar-host vulkan-stock:vulkan-stock:vulkan-linux vulkan-allreduce-host-f32 -
VIS=0,1,2,3 SCRIPT vk-27b-ar-off tools/lab/vulkan/probe-27b-vk.sh @b-vk-ar-host /mnt/data/bigcherry-work/runs/vk-27b-ar-off/out (vk-layer|vk-tensor|vk-tensor-mtp5) 3
VIS=0,1,2,3 SCRIPT vk-27b-ar-on tools/lab/vulkan/probe-27b-vk.sh @b-vk-ar-host /mnt/data/bigcherry-work/runs/vk-27b-ar-on/out (vk-layer|vk-tensor|vk-tensor-mtp5) 3 BIGCHERRY_VK_ALLREDUCE=host-f32 BIGCHERRY_PATCH_TRACE=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
