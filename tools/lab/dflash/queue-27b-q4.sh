#!/bin/bash
# 27B DFlash2 Q4_K_M vs Q8_0 bench (1286 build, dual XTX -sm tensor, draft on the R9700 unless noted):
# Q4 vs Q8 at n 7 in ABBA order, Q4 at n 4/10, on the 6900, and 3-card + 6900; 3 timed requests per layout.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT dflash-27b-q4 tools/lab/dflash/probe-27b.sh @b-27b-1286 /mnt/data/bigcherry-work/runs/dflash-27b-q4/out (plain|q4-.*) 3
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
