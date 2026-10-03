#!/bin/bash
# 1295 v2 quick screen on top of the deployment candidate (deployment draft settings in both arms), ~80K cached.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 QUICK_DEPTH=65536
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-gather-2 bigcherry:stock:linux-multi ar-cpu-root-kpool-topk-trim-gather gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-gather-v2-quick tools/lab/flash-next/quick-ab.sh $base @b-flash-gather-2 /mnt/data/bigcherry-work/runs/flashnext-gather-v2-quick
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
