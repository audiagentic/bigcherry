#!/bin/bash
# RCCL channel count (prefill is where RCCL wins; more channels may widen it, fewer may cut decode
# latency): default vs 2 vs >= 8 channels on the production binary, dual XTX 27B Q8_0 MTP.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
echo "AB ab-27b-rccl-channels tools/lab/native-vs-patched/server-ab-rccl-channels.json --pairs 6" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
