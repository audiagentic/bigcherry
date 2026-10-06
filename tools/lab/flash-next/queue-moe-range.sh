#!/bin/bash
# MET02 / 1281 phase A: build the production set (which carries 1281) and run the CPU reference test of
# ggml_mul_mat_id_range. No model, no GPU work beyond the build.
# Usage: queue-moe-range.sh <tag> [wait=<chain log with ALL_RUNS_DONE>]
set -u
TAG=${1:?tag}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_RUNS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
RUN=b-moerange-$TAG
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT moerange-$TAG-test tools/lab/flash-next/mmid-range-test.sh @$RUN $R/moerange-$TAG-test
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^rc=|^FAIL|test-mul-mat-id-range:|BUILD_FAILED|error" $R/moerange-$TAG-test.log | head -20
echo ALL_RUNS_DONE
