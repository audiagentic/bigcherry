#!/bin/bash
# 3-GPU (2x 7900 XTX + R9700) Qwen3.8-27B BF16 (~55 GB, needs all three cards): RCCL vs 1276
# adaptive N=3 (root3 below the 1 MiB switch, RCCL above) vs host (1244 root3 only), one
# multi-arch binary, no MTP. Needs the R9700 free: stops radiance-vllm and restarts it after.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf
L=tools/lab/native-vs-patched
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2 BUILD b-3g-nway bigcherry:stock:linux-multi ar-adaptive-nway gfx1100,gfx1201
VIS=0,1,2 AB ab-3g-bf16-nway $L/server-ab-3g-bf16-nway.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
