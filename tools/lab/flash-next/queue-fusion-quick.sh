#!/bin/bash
# Quick screens of existing fusion patches on the deployment candidate (deployment draft settings, ~30K cached).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-deploy-1205 bigcherry:stock:linux-multi deploy-plus-1205 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-q1205 tools/lab/flash-next/quick-ab.sh $base @b-deploy-1205 /mnt/data/bigcherry-work/runs/flashnext-q1205
VIS=0,1,2,3 BUILD b-deploy-1206 bigcherry:stock:linux-multi deploy-plus-1206 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-q1206 tools/lab/flash-next/quick-ab.sh $base @b-deploy-1206 /mnt/data/bigcherry-work/runs/flashnext-q1206
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
