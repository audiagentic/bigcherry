#!/bin/bash
# 1295 v2 full ABBA (10K/80K/160K) on the deployment candidate.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT flashnext-gather-v2-ab tools/lab/flash-next/gather-v2-ab.sh $base /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/7696d6ef1a1d3bf20f1141c3d1e6b9d8/ea3ea8de3f766008b0bb97931abdc141/bin/llama-server /mnt/data/bigcherry-work/runs/flashnext-gather-v2-ab
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
