#!/bin/bash
# 27B draft x deployment matrix on all four GPUs (3-card rows use the R9700): stops radiance-vllm for the
# run and restarts it (restart policy back to unless-stopped).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-27b-1286 draft-local-shared gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT dflash-27b-3 tools/lab/dflash/probe-27b.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/dflash-27b-3/out (plain|mtp|dflash|dspark).*
VIS=0,1,2,3 SCRIPT dflash-27b-3-1286 tools/lab/dflash/probe-27b.sh @b-27b-1286 /mnt/data/bigcherry-work/runs/dflash-27b-3-1286/out (plain|p1286-.*)
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
