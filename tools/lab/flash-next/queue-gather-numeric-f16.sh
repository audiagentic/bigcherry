#!/bin/bash
# 1295 numeric check with f16 target KV: both paths then use f16 attention math (the masked path's q8_0 vector
# kernel quantizes Q to q8_1), so remaining probability differences isolate the gather itself.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export CTK=f16 CTV=f16 DEPTHS=8192 CTX=65536
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT flashnext-gather-numeric-f16 tools/lab/flash-next/gather-numeric.sh /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/cb765fc5ba3f516f1cfd5d170b18642a/3a9a890b50c7321e7d114e03fc8b2aaa/bin/llama-server /mnt/data/bigcherry-work/runs/flashnext-gather-numeric-f16" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
