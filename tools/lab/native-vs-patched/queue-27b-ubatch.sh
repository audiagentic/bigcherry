#!/bin/bash
# Prefill: ubatch 512 (production) vs 1024 vs 2048 (batch stays 2048) on the adaptive binary,
# 27B Q8_0 dual XTX, MTP depth 5. Larger ubatch = fewer, larger AllReduces and larger matmuls.
# Runtime setting only; no rebuild.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl adaptive-wire-latency
AB ab-27b-ubatch $L/server-ab-ubatch.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py acceptance /mnt/data/bigcherry-work/runs/ab-27b-ubatch/result
rm -f "$jobs"
echo ALL_JOBS_DONE
