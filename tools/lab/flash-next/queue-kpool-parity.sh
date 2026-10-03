#!/bin/bash
# Queue the 1292 no-MTP greedy parity.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
w=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds
base=$w/1356624955c2dd053598169ca58e5392/774ea136f428afd311569c9c47387042/bin/llama-server
new=$w/5b196af232266bae5856251b03acfe28/14bcf2379b53a5114df9f4928a18d202/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT flashnext-kpool-parity-1 tools/lab/flash-next/kpool-parity.sh $base $new /mnt/data/bigcherry-work/runs/flashnext-kpool-parity-1" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
