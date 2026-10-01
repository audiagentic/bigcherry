#!/bin/bash
# PGC09 investigation: plain decode (no MTP) on the contract session's control (validated BC + 0860 +
# 1225, RCCL) vs subject (+ 0840, adaptive) builds. MTP checks showed equal decode (43.4 vs 43.3 t/s)
# although the stock-lane A/B gave +7%; this tells whether the validated set cancels the gain.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
L=tools/lab/native-vs-patched
B=/mnt/data/bigcherry-work/builds/gfx1100-974f5feb
export SPEC_ARGS= EXTRA_ARGS="--flash-attn on -c 8192"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
SCRIPT pgc09-lane-check-3 $L/pgc09-lane-check.sh $B/251b4293c9d2af009fa0055b8ae41e3cf2b3aec230313c8115ea8e5998c06aa5/control/bin $B/df1d665201ff7ba473bf97c7da40b3f43614a7189cba069cdd111c4c58de4cc8/validation-subject/bin /mnt/data/bigcherry-work/runs/pgc09-lane-check-3/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
