#!/bin/bash
# MTP greedy parity on the t-0840d builds: control (RCCL) and subject (adaptive, f32 host).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
B=/mnt/data/bigcherry-work/builds/gfx1100-974f5feb
jobs=$(mktemp)
cat > "$jobs" <<JOBS
SCRIPT mtp-parity-control-1 $L/mtp-greedy-parity.sh $B/251b4293c9d2af009fa0055b8ae41e3cf2b3aec230313c8115ea8e5998c06aa5/control/bin /mnt/data/bigcherry-work/runs/mtp-parity-control-1/out
SCRIPT mtp-parity-subject-1 $L/mtp-greedy-parity.sh $B/028bfab7a7e1dc25c4dc16daf1241acf1186ff7f311e8b201a9bf38513053d8a/validation-subject/bin /mnt/data/bigcherry-work/runs/mtp-parity-subject-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
