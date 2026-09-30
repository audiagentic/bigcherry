#!/bin/bash
# 1275 small-AllReduce latency switches on the internal (host) provider, dual XTX 27B Q8_0 MTP:
# pristine vs BIGCHERRY_AR_SLOT_SYNC=none vs BIGCHERRY_AR_SMALL_BLOCKS=1, one binary. The preflight
# exports the switches so the once-per-process marker (with the chosen values) must appear.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export GGML_CUDA_ALLREDUCE=internal BIGCHERRY_AR_SLOT_SYNC=none BIGCHERRY_AR_SMALL_BLOCKS=1
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-ar-small ar-small-latency
PREFLIGHT pf-27b-ar-small @b-27b-ar-small $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1275_ar_small.*blocks=1.*slot_sync=none $SRV
REQUIRES=pf-27b-ar-small AB ab-27b-ar-small $L/server-ab-ar-small.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
