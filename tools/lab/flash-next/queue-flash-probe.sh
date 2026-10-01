#!/bin/bash
# QFN01 Flash-Next layout probe on all four GPUs. Stops radiance-vllm (R9700) for the run and restarts it
# (restart policy back to unless-stopped). Launch behind a waiter so vLLM is only down while this runs.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-4g-ts qwen4exp-tensor gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-probe-2 tools/lab/flash-next/layout-probe.sh @b-flash-4g-ts /mnt/data/bigcherry-work/runs/flashnext-probe-2/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
