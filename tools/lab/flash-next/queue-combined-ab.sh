#!/bin/bash
# Build the deployment candidate (1291+1292+1294+1297) and run the end-to-end combined A/B vs this morning's build.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1356624955c2dd053598169ca58e5392/774ea136f428afd311569c9c47387042/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-deploy-1 bigcherry:stock:linux-multi ar-cpu-root-kpool-topk-trim gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-combined-ab-1 tools/lab/flash-next/combined-ab.sh $base @b-flash-deploy-1 /mnt/data/bigcherry-work/runs/flashnext-combined-ab-1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
