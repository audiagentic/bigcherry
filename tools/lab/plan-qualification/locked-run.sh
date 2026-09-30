#!/bin/bash
# Run one command holding the queue's host-exclusive activity lock and the locks for GPUs ${BC_GPUS:-0,1},
# so builds and balanced A/Bs never overlap a timed campaign.
# Usage: locked-run.sh <command> [args...]
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
source "$here/work-root.sh"
work=$(work_root_resolve "$root")
activity_lock_exclusive_acquire "$work"
gpu_lock_acquire "$work" "${BC_GPUS:-0,1}"
cd "$root"
"$@"
