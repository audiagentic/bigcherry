#!/bin/bash
# MTP draft depth sweep (config only) on the production binary, dual XTX 27B Q8_0 -sm tensor:
# --spec-draft-n-max 2/3/4 and 4/5/6 (4 = current default, shared anchor). Per-arm server_args
# are appended after the shared args, so they override its --spec-draft-n-max 4. Acceptance per arm
# differs by design here (it is the mechanism); throughput decides.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
AB ab-27b-mtp-depth-a $L/server-ab-mtp-depth-a.json --pairs 6
AB ab-27b-mtp-depth-b $L/server-ab-mtp-depth-b.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
