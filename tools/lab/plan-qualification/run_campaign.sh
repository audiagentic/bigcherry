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
work=${BIGCHERRY_WORK_ROOT:-$root/work}
: "${BC_HIP_PATH:?set BC_HIP_PATH}" "${BC_MODEL:?set BC_MODEL}"
mkdir -p "$work/tmp"
export TMPDIR=$work/tmp
export PYTHONPATH=tools ROCM_PATH=$BC_HIP_PATH HIP_PATH=$BC_HIP_PATH PATH=$BC_HIP_PATH/bin:$PATH
args=(--patch "$patch" --baseline-source bigcherry-tuning --amdgpu-targets "$arch"
      --device-map "$arch=$dev" --model "$BC_MODEL" --hip-path "$BC_HIP_PATH"
      --workdir "$work/runs/$run" --worktree-root "$work/worktrees"
      --build-root "$work/builds/$patch-$arch")
[ "$producer" != "-" ] && args+=(--validation-producer "$producer")
python3 -m bigcherry.patch.validation_campaign "${args[@]}" "$@"
echo "CAMPAIGN_EXIT=$?"
