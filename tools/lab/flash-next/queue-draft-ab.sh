#!/bin/bash
# Draft investigation: QSA attention scaling (10K decode profile) + draft KV q8_0 vs f16.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
new=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/090aa314b2d5e250727d572d7d80441e/841024fa50a66cd35746f80486cb6c43/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT flashnext-draft-ab-1 tools/lab/flash-next/draft-ab.sh $new /mnt/data/bigcherry-work/runs/flashnext-draft-ab-1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
