#!/bin/bash
# 27B MTP vs DFlash probe (dual XTX + 6900 draft). The R9700 is not used, so radiance-vllm keeps running.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,3 SCRIPT dflash-27b-1 tools/lab/dflash/probe-27b.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/dflash-27b-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
