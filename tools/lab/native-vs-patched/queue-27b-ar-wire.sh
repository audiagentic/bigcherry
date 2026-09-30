#!/bin/bash
# 1272 host-AllReduce wire sweep on dual XTX. GGML_CUDA_AR_WIRE unset is the
# pristine path and emits no 1272 marker. PREFLIGHT cannot set per-run env yet,
# so this queue is BUILD -> AB; each A/B arm supplies its explicit wire env.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-ar-wire ar-wire
AB ab-27b-ar-wire $L/server-ab-ar-wire.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
