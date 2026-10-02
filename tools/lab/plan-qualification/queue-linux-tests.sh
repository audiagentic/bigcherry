#!/bin/bash
# Queue the Linux full test suite (no GPU work; host lock only so it never perturbs a measurement).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
# queue.sh requires a model and HIP path even for script-only rows; the tests use neither.
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
run=linux-tests-$(date +%Y%m%dT%H%M)
cat > "$jobs" <<JOBS
VIS=0 SCRIPT $run tools/lab/plan-qualification/linux-test-suite.sh /mnt/data/bigcherry-work/test-clone patch-refactor
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
