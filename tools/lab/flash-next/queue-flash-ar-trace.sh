#!/bin/bash
# Build the 1277 trace experiment on the production composition and run the Flash-Next AllReduce census.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-ar-trace bigcherry:stock:linux-multi adaptive-size-trace gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-ar-trace tools/lab/flash-next/ar-trace.sh @b-flash-ar-trace /mnt/data/bigcherry-work/runs/flashnext-ar-trace/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
