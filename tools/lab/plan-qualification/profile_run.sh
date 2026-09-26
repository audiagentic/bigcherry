#!/bin/bash
# PVPS10 kernel-coverage profile of a patch (control vs subject) on one workload.
# Builds any missing tree through the scaffold's shared per-arch+toolchain build
# root (control is a no-op after the first job), then rocprofv3 + kernel-fraction.
#
# Usage: profile_run.sh <patch-id> <arch> <device> <prefill|decode> <run-name> [extra args...]
#   extra args pass through to bigcherry.patch.campaign.profile
#   (e.g. --common-patches a,b  --env GGML_CUDA_DQ_MMV=1)
# Host paths come from the environment as for run_campaign.sh (BC_MODEL,
# BC_HIP_PATH). Output: <work>/runs/<run-name>/ ; ends with PROFILE_EXIT=<rc>.
set -u
patch=$1; arch=$2; dev=$3; workload=$4; run=$5; shift 5
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
: "${BC_HIP_PATH:?set BC_HIP_PATH}" "${BC_MODEL:?set BC_MODEL}"
mkdir -p "$work/tmp"
export TMPDIR=$work/tmp
export CCACHE_DIR=${CCACHE_DIR:-$work/ccache} CCACHE_BASEDIR=$work CCACHE_NOHASHDIR=1
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
export PYTHONPATH=tools ROCM_PATH=$BC_HIP_PATH HIP_PATH=$BC_HIP_PATH PATH=$BC_HIP_PATH/bin:$PATH
toolchain=$(printf '%s' "$BC_HIP_PATH" | sha256sum | cut -c1-8)
python3 -m bigcherry.patch.campaign.profile --patch "$patch" --arch "$arch" --device "$dev" \
  --model "$BC_MODEL" --workload "$workload" --hip-path "$BC_HIP_PATH" \
  --worktree-root "$work/worktrees" --build-root "$work/builds/$arch-$toolchain" \
  --out "$work/runs/$run" "$@"
echo "PROFILE_EXIT=$?"
