#!/bin/bash
# QFN01 Flash-Next AllReduce x topology sweep on the production build: provider auto/ccl/host/adaptive
# (switch 32K/96K/256K/1M)/butterfly at the best config (-ts 4,4,3, ub1024, MTP3 draft on the 6900), plus
# 4-card tensor including the 6900 (host provider: the 6900 cannot join RCCL) vs the 3-card reference.
# Greedy output of each MTP arm is checked against the 3-card baseline. Stops/restarts radiance-vllm.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT flashnext-ar-1 tools/lab/flash-next/layout-probe.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/flashnext-ar-1/out (cpu3-tensor|ar-.*)
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
