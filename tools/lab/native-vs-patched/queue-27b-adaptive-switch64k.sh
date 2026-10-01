#!/bin/bash
# Diagnose adaptive's pp1024 -3.9% (plain decode, ubatch 512): RCCL vs adaptive (1 MiB switch)
# vs adaptive with a 64 KiB switch (decode ARs are 20 KB, the 4-token prompt-tail ARs 80 KB;. If the pp1024
# see ar-size-trace-1). If pp1024 recovers while decode keeps its gain, the tail on the host path
# is the cause and a 64 KiB switch is a candidate fix (plain decode; MTP verify ARs are ~100-120 KB).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl adaptive-wire-latency
AB ab-27b-adaptive-switch64k $L/server-ab-adaptive-switch64k.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
