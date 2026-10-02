#!/bin/bash
# Queue the 27B Q8_0 dual-gfx1100 rocprofv3 attribution run on the adaptive-wire-latency binary.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl bigcherry:stock:linux-multi adaptive-wire-latency gfx1100
SCRIPT prof-27b-q8-2 tools/lab/profiling/profile-27b-q8.sh @b-27b-awl /mnt/data/bigcherry-work/runs/prof-27b-q8-2/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
