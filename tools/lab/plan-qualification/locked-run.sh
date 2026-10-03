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
# A GPU root-port link throttled below its max speed (thermald driving the PCIe_Port_Link_Speed cooling
# devices) halves AllReduce-bound prefill; refuse to run rather than record degraded numbers.
bash "$here/pcie-link-check.sh" || { echo "PCIE_LINK_DEGRADED: refusing to run $*" >&2; exit 97; }
# Same shared compiler cache as run_campaign.sh: content-addressed worktrees and build dirs
# differ only by path, so BASEDIR + NOHASHDIR let every build reuse identical objects.
export CCACHE_DIR=${CCACHE_DIR:-$work/ccache} CCACHE_BASEDIR=$work CCACHE_NOHASHDIR=1
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
cd "$root"
"$@"
