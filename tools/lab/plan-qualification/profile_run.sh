#!/bin/bash
# PVPS10 kernel-coverage profile of a patch (control vs subject) on one workload.
#
# BC_PROFILE_PHASE controls queue scheduling:
#   prepare  - serialize shared CMake/Ninja mutation, write prepared manifest
#   run      - verify prepared manifest and profile under a shared build lock
#   both     - compatibility/direct mode; prepare then run
#
# Usage: profile_run.sh <patch-id> <arch> <device> <prefill|decode> <run-name> [extra args...]
set -u
patch=$1; arch=$2; dev=$3; workload=$4; run=$5; shift 5
phase=${BC_PROFILE_PHASE:-both}
case "$phase" in prepare|run|both) ;; *) echo "invalid BC_PROFILE_PHASE=$phase" >&2; exit 2 ;; esac
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
build_lock="$work/queue/build-locks/$arch-$toolchain.lock"
source "$root/tools/lab/plan-qualification/gpu-lock.sh"
source "$root/tools/lab/plan-qualification/activity-lock.sh"

prepare_profile() {
    local build_fd rc
    # Exclusive build-key ownership protects the shared CMake/Ninja tree.
    # Acquire build-key before host activity: jobs waiting for the same tree do
    # not consume host-activity capacity or delay a pending performance writer.
    exec {build_fd}>"$build_lock"
    flock -x "$build_fd"
    activity_lock_shared_acquire "$work"
    rm -f "$prepared"
    python3 -m bigcherry.patch.campaign.profile --patch "$patch" --arch "$arch" --device "$dev" \
      --model "$BC_MODEL" --workload "$workload" --hip-path "$BC_HIP_PATH" \
      --worktree-root "$work/worktrees" --build-root "$work/builds/$arch-$toolchain" \
      --out "$work/runs/$run" --prepare-only --prepared-manifest "$prepared" "$@"
    rc=$?
    activity_lock_release
    eval "exec $build_fd>&-"
    echo "PROFILE_PREPARE_EXIT=$rc"
    return "$rc"
}

run_profile() {
    local build_fd rc
    [ -f "$prepared" ] || { echo "prepared profile manifest missing: $prepared" >&2; return 2; }
    # Shared build-key ownership allows any number of same-build traces while
    # preventing a CMake/Ninja preparer from changing their executable/libs.
    exec {build_fd}>"$build_lock"
    flock -s "$build_fd"
    activity_lock_shared_acquire "$work"
    gpu_lock_acquire "$work" "$dev"
    python3 -m bigcherry.patch.campaign.profile --patch "$patch" --arch "$arch" --device "$dev" \
      --model "$BC_MODEL" --workload "$workload" --hip-path "$BC_HIP_PATH" \
      --worktree-root "$work/worktrees" --build-root "$work/builds/$arch-$toolchain" \
      --out "$work/runs/$run" --prepared-manifest "$prepared" "$@"
    rc=$?
    echo "PROFILE_EXIT=$rc"
    return "$rc"
}

if [ "$phase" = prepare ]; then
    prepare_profile "$@"
    exit $?
fi
if [ "$phase" = run ]; then
    run_profile "$@"
    exit $?
fi
prepare_profile "$@" || exit $?
run_profile "$@"
exit $?
