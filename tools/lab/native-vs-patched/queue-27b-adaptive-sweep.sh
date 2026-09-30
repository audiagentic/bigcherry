#!/bin/bash
# 0840 adaptive (host below --allreduce-switch-bytes, RCCL at/above) with 1272 composed, dual XTX
# 27B Q8_0 MTP: sweep the crossover at 256 KiB / 1 MiB / 4 MiB on one binary. Decode reductions
# are ~20-100 KiB, 512-token prefill reductions ~10 MiB F32, so all three should route decode to
# host and prefill to RCCL; the sweep checks where mid-size (ubatch tail / MTP verify) should go.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-adaptive-wire allreduce-adaptive-wire
AB ab-27b-adaptive-sweep $L/server-ab-adaptive-sweep.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
