#!/bin/bash
# PGC09 investigation (t-0840d RCCL builds, MTP depth 5, production flags, both arms forced to RCCL (--allreduce ccl): does the subject build itself change acceptance?) on the contract session's control (validated BC + 0860 +
# 1225, RCCL) vs subject (+ 0840, adaptive) builds. MTP checks showed equal decode (43.4 vs 43.3 t/s)
# although the stock-lane A/B gave +7%; this tells whether the validated set cancels the gain.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
L=tools/lab/native-vs-patched
B=/mnt/data/bigcherry-work/builds/gfx1100-974f5feb
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 5" EXTRA_ARGS="--allreduce ccl --flash-attn on -c 64000 --ubatch-size 2048 --batch-size 2048 --threads 8 -ctkd q8_0 -ctvd q8_0"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
SCRIPT pgc09-lane-check-7 $L/pgc09-lane-check.sh $B/251b4293c9d2af009fa0055b8ae41e3cf2b3aec230313c8115ea8e5998c06aa5/control/bin $B/028bfab7a7e1dc25c4dc16daf1241acf1186ff7f311e8b201a9bf38513053d8a/validation-subject/bin /mnt/data/bigcherry-work/runs/pgc09-lane-check-7/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
