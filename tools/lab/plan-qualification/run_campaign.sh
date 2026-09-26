#!/bin/bash
# Run one patch validation campaign on the build server.
# Usage: run_campaign.sh <patch-id> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# Host specifics come from the environment (never hardcoded here):
#   BC_HIP_PATH  ROCm prefix the campaign compiles with (clang wrapper prefix)
#   BC_MODEL     model gguf for the producer/benchmark
# Output: work/runs/<run-name>/ (campaign workdir) and work/runs/<run-name>.log
set -u
patch=$1; producer=$2; arch=$3; dev=$4; run=$5; shift 5
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
: "${BC_HIP_PATH:?set BC_HIP_PATH}" "${BC_MODEL:?set BC_MODEL}"
mkdir -p "$work/tmp"
export TMPDIR=$work/tmp
# Shared compiler cache for every campaign tree. Worktrees and build dirs have
# content-addressed paths, so BASEDIR (relative paths) + NOHASHDIR are what let
# control/subject/stock/other sessions reuse identical objects; the cache is
# sized for HIP objects (the 5 GiB default thrashes).
export CCACHE_DIR=${CCACHE_DIR:-$work/ccache} CCACHE_BASEDIR=$work CCACHE_NOHASHDIR=1
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
mkdir -p "$CCACHE_DIR"
export PYTHONPATH=tools ROCM_PATH=$BC_HIP_PATH HIP_PATH=$BC_HIP_PATH PATH=$BC_HIP_PATH/bin:$PATH
# One build root per architecture, shared by every patch: the scaffold keys
# each tree by its content-addressed source name (stock, base, validated
# control, and each subject's own trees), so trees that do not depend on the
# patch (stock/base/control) are built once per arch and reused by every job.
# Keyed by toolchain too, so alternating ROCm versions do not force reconfigures.
toolchain=$(printf '%s' "$BC_HIP_PATH" | sha256sum | cut -c1-8)
args=(--patch "$patch" --baseline-source bigcherry-tuning --amdgpu-targets "$arch"
      --device-map "$arch=$dev" --model "$BC_MODEL" --hip-path "$BC_HIP_PATH"
      --workdir "$work/runs/$run" --worktree-root "$work/worktrees"
      --build-root "$work/builds/$arch-$toolchain")
[ "$producer" != "-" ] && args+=(--validation-producer "$producer")
# A checkout whose patch catalog does not load (e.g. a patch naming a contract
# that was never added) fails every job in seconds and drains the whole queue.
# Wait for a loadable catalog instead, so the lane pauses until it is fixed.
until python3 -c "from bigcherry.patch import patchset; patchset.catalog()" 2>/dev/null; do
    echo "patch catalog does not load; waiting for a fixed checkout ($(date -Is))"
    sleep 300
done
python3 -m bigcherry.patch.validation_campaign "${args[@]}" "$@"
echo "CAMPAIGN_EXIT=$?"
