#!/bin/bash
# Adaptive AllReduce host wire f32 vs bf16 by decode mode, on the existing adaptive-wire-latency
# binary (no rebuild): plain decode (no MTP) and MTP draft depth 5 (27B production setting).
# Question: is the best decode wire mode-dependent (bf16 for plain, f32 for MTP)?
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl bigcherry:stock:linux-multi adaptive-wire-latency gfx1100
AB ab-27b-wire-plain $L/server-ab-wire-plain.json --pairs 4
AB ab-27b-wire-d5 $L/server-ab-wire-d5.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py acceptance /mnt/data/bigcherry-work/runs/ab-27b-wire-d5/result
rm -f "$jobs"
echo ALL_JOBS_DONE
