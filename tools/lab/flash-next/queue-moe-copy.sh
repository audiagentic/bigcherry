#!/bin/bash
# MET01 / 1336: build the production set + 1336 (experiment moe-copy-callback) and run moe-copy-ab.sh on the R9700
# and on one 7900 XTX (single GPU, routed experts in host memory).
# SUFFIX names a rerun of the script on the same build. EXPERIMENT=moe-expert-cache with ARMS=cache builds 1336 + 1337
# and runs the cache lanes (moe-copy-ab.sh reads ARMS, CACHE_MIB, NCMOE, GPU from the environment).
# Usage: queue-moe-copy.sh <tag> [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-${EXPERIMENT:-moecopy}-$TAG
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi ${EXPERIMENT:-moe-copy-callback} gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT moecopy-$TAG-r9700${SUFFIX:-} tools/lab/flash-next/moe-copy-ab.sh @$RUN $R/moecopy-$TAG-r9700${SUFFIX:-}
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^[A-Z0-9]+ (short|long)|^[A-Z0-9]+: |identity|^ +[0-9]+ |per request|^  (short|long)|SERVER_FAILED|BUILD_FAILED" $R/moecopy-$TAG-r9700${SUFFIX:-}.log | cut -c1-260
echo ALL_JOBS_DONE
