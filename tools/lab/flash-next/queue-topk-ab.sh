#!/bin/bash
# Build 1291+1292+1294 and run the 1294 A/B against the 1291+1292 build.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/5b196af232266bae5856251b03acfe28/14bcf2379b53a5114df9f4928a18d202/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-topk-1 bigcherry:stock:linux-multi ar-cpu-root-kpool-topk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-topk-ab-1 tools/lab/flash-next/topk-ab.sh $base @b-flash-topk-1 /mnt/data/bigcherry-work/runs/flashnext-topk-ab-1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
