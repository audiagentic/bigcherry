#!/bin/bash
# Diagnose adaptive's pp1024 -3.9% (plain decode, ubatch 512): RCCL vs adaptive (1 MiB switch)
# vs adaptive with a 16 KiB switch (below the ~20 KB per-token decode AllReduce). If the pp1024
# loss disappears at 16 KiB while decode keeps its gain, small non-decode AllReduces during
# prefill are taking the host path.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl bigcherry:stock:linux-multi adaptive-wire-latency gfx1100
AB ab-27b-adaptive-switch $L/server-ab-adaptive-switch.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
