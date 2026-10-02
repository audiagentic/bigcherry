#!/bin/bash
# 27B draft x deployment SCREENING (one representative per draft x deployment, 1 timed request) on all four GPUs;
# the detail runs (depth/quant variants, repeats) are queued separately for what survives.
# (3-card rows use the R9700): stops radiance-vllm for the
# run and restarts it (restart policy back to unless-stopped).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-27b-1286 bigcherry:stock:linux-multi draft-local-shared gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT dflash-27b-screen tools/lab/dflash/probe-27b.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/dflash-27b-screen/out (plain|mtp5|mtpside5-d6900|mtpside5-dR9700|plain-3card|mtp5-3card|plain-layer|mtp5-layer|dflash-q8-n7-layer-d6900|dspark-q8-n6-layer-d6900) 1
VIS=0,1,2,3 SCRIPT dflash-27b-screen-1286 tools/lab/dflash/probe-27b.sh @b-27b-1286 /mnt/data/bigcherry-work/runs/dflash-27b-screen-1286/out (plain|p1286-mtp5|p1286-dflash-q8-n7-d6900|p1286-dflash-q8-n7-dR9700|p1286-dspark-q8-n6-d6900) 1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
