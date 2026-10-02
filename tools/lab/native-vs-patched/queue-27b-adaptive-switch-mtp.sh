#!/bin/bash
# Adaptive switch threshold under MTP (27B Q8_0 dual XTX, draft depth 5, ubatch 2048): RCCL vs
# adaptive 1 MiB vs adaptive 64 KiB. 64 KiB fixes the plain-decode pp1024 loss (4-token prompt
# tail ARs are 80 KB), but MTP verify ARs (~100-120 KB) also move to RCCL; this measures the cost.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl bigcherry:stock:linux-multi adaptive-wire-latency gfx1100
AB ab-27b-adaptive-switch-mtp $L/server-ab-adaptive-switch-mtp.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py acceptance /mnt/data/bigcherry-work/runs/ab-27b-adaptive-switch-mtp/result
rm -f "$jobs"
echo ALL_JOBS_DONE
