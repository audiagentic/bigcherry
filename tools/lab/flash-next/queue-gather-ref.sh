#!/bin/bash
# 1295 accuracy vs f32 CPU reference (A/B kept from run 1, C rerun on CPU): masked FA vs gathered FA.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export CTX=65536 ONLY_REF=1
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT flashnext-gather-ref-cpu tools/lab/flash-next/gather-ref.sh /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/cb765fc5ba3f516f1cfd5d170b18642a/3a9a890b50c7321e7d114e03fc8b2aaa/bin/llama-server /mnt/data/bigcherry-work/runs/flashnext-gather-ref-1" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
