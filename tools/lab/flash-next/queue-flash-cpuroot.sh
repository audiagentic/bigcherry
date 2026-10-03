#!/bin/bash
# Goal (Flash-Next AllReduce): build the production composition + 1291 (--allreduce cpu-root) and A/B it
# against auto (RCCL on 3 GPUs) on Flash-Next -ts 4,4,3 ub1024, no MTP and MTP3 (draft on the 6900), ABBA,
# with BIGCHERRY_PATCH_TRACE for the activation marker. Greedy output checked against cr-nomtp-auto-a.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BIGCHERRY_PATCH_TRACE=1
export BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=33554432
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-cpuroot-6 bigcherry:stock:linux-multi ar-cpu-root gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-cpuroot-6 tools/lab/flash-next/layout-probe.sh @b-flash-cpuroot-6 /mnt/data/bigcherry-work/runs/flashnext-cpuroot-6/out cr-.*
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
