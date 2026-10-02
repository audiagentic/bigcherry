#!/bin/bash
# Queue the 27B prefill ubatch check on dual XTX (R9700/vLLM untouched): stock c061 build vs the
# adaptive-patched 27B build.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1 SCRIPT pp-ubatch-check-1 tools/lab/native-vs-patched/pp-ubatch-check.sh /mnt/data/bigcherry-work/runs/pp-ubatch-check-1/out @b-flash-c061 @b-27b-awl
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
