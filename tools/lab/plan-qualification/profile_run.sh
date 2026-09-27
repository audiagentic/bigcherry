#!/bin/bash
# PVPS10 kernel-coverage profile of a patch (control vs subject) on one workload.
# Shared build preparation is fenced by architecture+toolchain, then released
# before GPU profiling so prepared profiles can overlap on different devices.
#
# Usage: profile_run.sh <patch-id> <arch> <device> <prefill|decode> <run-name> [extra args...]
set -u
patch=$1; arch=$2; dev=$3; workload=$4; run=$5; shift 5
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
: "${BC_HIP_PATH:?set BC_HIP_PATH}" "${BC_MODEL:?set BC_MODEL}"
mkdir -p "$work/tmp" "$work/runs/$run" "$work/queue/build-locks"
export TMPDIR=$work/tmp
export CCACHE_DIR=${CCACHE_DIR:-$work/ccache} CCACHE_BASEDIR=$work CCACHE_NOHASHDIR=1
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
export PYTHONPATH=tools ROCM_PATH=$BC_HIP_PATH HIP_PATH=$BC_HIP_PATH PATH=$BC_HIP_PATH/bin:$PATH
toolchain=$(printf '%s' "$BC_HIP_PATH" | sha256sum | cut -c1-8)
prepared="$work/runs/$run/prepared-profile.json"

# Phase 1: shared build preparation. Same arch+toolchain jobs serialize only
# while CMake/Ninja may mutate the common build tree. The resulting manifest
# hashes both binaries, so the GPU phase fails closed if anything changes.
exec {build_fd}>"$work/queue/build-locks/$arch-$toolchain.lock"
flock "$build_fd"
python3 -m bigcherry.patch.campaign.profile --patch "$patch" --arch "$arch" --device "$dev" \
  --model "$BC_MODEL" --workload "$workload" --hip-path "$BC_HIP_PATH" \
  --worktree-root "$work/worktrees" --build-root "$work/builds/$arch-$toolchain" \
  --out "$work/runs/$run" --prepare-only --prepared-manifest "$prepared" "$@"
rc=$?
eval "exec $build_fd>&-"
if [ "$rc" -ne 0 ]; then
    echo "PROFILE_EXIT=$rc"
    exit "$rc"
fi

# Phase 2: non-performance GPU trace. Multiple prepared profiles may overlap on
# different GPUs. The host shared gate blocks if a timed campaign is pending or
# active; the per-device lock prevents same-card collisions.
source "$root/tools/lab/plan-qualification/gpu-lock.sh"
source "$root/tools/lab/plan-qualification/activity-lock.sh"
activity_lock_shared_acquire "$work"
gpu_lock_acquire "$work" "$dev"
python3 -m bigcherry.patch.campaign.profile --patch "$patch" --arch "$arch" --device "$dev" \
  --model "$BC_MODEL" --workload "$workload" --hip-path "$BC_HIP_PATH" \
  --worktree-root "$work/worktrees" --build-root "$work/builds/$arch-$toolchain" \
  --out "$work/runs/$run" --prepared-manifest "$prepared" "$@"
rc=$?
echo "PROFILE_EXIT=$rc"
exit "$rc"
