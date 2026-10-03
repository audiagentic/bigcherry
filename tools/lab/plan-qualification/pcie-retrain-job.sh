#!/bin/bash
# Restore degraded GPU root-port links with every GPU idle: takes the queue's host-exclusive lock and all
# four GPU locks (directly, not via locked-run.sh, whose link preflight would refuse), stops radiance-vllm
# (R9700), runs pcie-link-check.sh --retrain (sudo password on stdin), restarts radiance-vllm with restart
# policy unless-stopped. Usage: pcie-retrain-job.sh < password
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
here=tools/lab/plan-qualification
pw=$(cat)
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null && echo "VLLM_STOPPED $(date -Is)"
source "$here/work-root.sh"
work=$(work_root_resolve "$(pwd)")
activity_lock_exclusive_acquire "$work"
gpu_lock_acquire "$work" 0,1,2,3
printf '%s\n' "$pw" | bash "$here/pcie-link-check.sh" --retrain
rc=$?
echo "RETRAIN_EXIT=$rc"
exit $rc
