#!/bin/bash
# Queue the 1291 long-context fit sweep on the b-flash-cpuroot-6 binary.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
bin=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1356624955c2dd053598169ca58e5392/774ea136f428afd311569c9c47387042/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT flashnext-long-ctx-fit-1 tools/lab/flash-next/long-ctx-fit.sh $bin /mnt/data/bigcherry-work/runs/flashnext-long-ctx-fit-1" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
