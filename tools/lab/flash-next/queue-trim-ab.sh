#!/bin/bash
# Build the 1297 stack and A/B the trimmed draft head by env (one binary).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/090aa314b2d5e250727d572d7d80441e/841024fa50a66cd35746f80486cb6c43/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT flashnext-trim-quick-3 tools/lab/flash-next/quick-ab.sh /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/cb765fc5ba3f516f1cfd5d170b18642a/3a9a890b50c7321e7d114e03fc8b2aaa/bin/llama-server /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/cb765fc5ba3f516f1cfd5d170b18642a/3a9a890b50c7321e7d114e03fc8b2aaa/bin/llama-server /mnt/data/bigcherry-work/runs/flashnext-trim-quick-3 BIGCHERRY_DRAFT_VOCAB_N=65536
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
