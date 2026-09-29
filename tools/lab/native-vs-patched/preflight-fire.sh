#!/usr/bin/env bash
# Pre-flight "does this patch fire on this model" gate. Run BEFORE any timed A/B for a patch/model pair.
# Usage: preflight-fire.sh BINARY MODEL PATTERN [extra server args...]
# Takes the queue's host-exclusive + per-GPU locks (so it never perturbs a running campaign), then runs
# activation-check.sh (traced short completion on GPUs 0,1). Exit 0 only if PATTERN was hit at least once.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
source "$root/tools/lab/plan-qualification/work-root.sh"
work=$(work_root_resolve "$root")
activity_lock_exclusive_acquire "$work"
gpu_lock_acquire "$work" 0
gpu_lock_acquire "$work" 1
out=$(bash "$here/activation-check.sh" "$@")
echo "$out"
hits=$(echo "$out" | sed -n 's/^hits: //p')
[ "${hits:-0}" -ge 1 ]
