#!/bin/bash
# 27B DETAIL runs for what survived screening (2026-10-02): 3-card MTP depth 4/5/6 + sidecar/DFlash on the
# 6900, dual-XTX DFlash2 (1286) on the R9700 at n 4/7/10 and Q4_K_M, with A/B repeats of the leaders;
# 3 timed requests per layout. Stops radiance-vllm for the run and restarts it.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT dflash-27b-detail tools/lab/dflash/probe-27b.sh @b-27b-1286 /mnt/data/bigcherry-work/runs/dflash-27b-detail/out (plain|dt-.*) 3
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
