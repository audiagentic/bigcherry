#!/bin/bash
# 1292 kpool tail truncation: build 1291+1292, then ABBA decode at ~80K cached context (MTP3, q8_0 KV)
# against the 1291-only b-flash-cpuroot-6 binary.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1356624955c2dd053598169ca58e5392/774ea136f428afd311569c9c47387042/bin/llama-server
r=/mnt/data/bigcherry-work/runs/flashnext-kpool-ab-1
s=tools/lab/flash-next/long-ctx-profile.sh
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-kpool-1 bigcherry:stock:linux-multi ar-cpu-root-kpool gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT kpool-base-a $s $base $r/base-a timing
VIS=0,1,2,3 SCRIPT kpool-new-a $s @b-flash-kpool-1 $r/new-a timing
VIS=0,1,2,3 SCRIPT kpool-new-b $s @b-flash-kpool-1 $r/new-b timing
VIS=0,1,2,3 SCRIPT kpool-base-b $s $base $r/base-b timing
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
