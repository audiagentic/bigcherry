#!/bin/bash
# Kernel-level profile of a finished campaign run's own control and subject
# binaries (PVPS10 audit): which kernels ran, how often, and their GPU-time
# share, so a verdict can be checked against what the patch actually changed.
#
# Usage: profile_run.sh <arch> <device> <run-name> <workload> [ENV=VAL ...]
#   workload: prefill (llama-bench -p 512 -n 0) or decode (-p 0 -n 128)
#   ENV=VAL : extra environment for both arms (e.g. a patch's opt-in switch)
# Host paths come from the environment as for run_campaign.sh (BC_MODEL,
# BC_HIP_PATH); binaries are located from the run's own log (control/subject
# source hashes) under the shared per-arch+toolchain build root.
# Output: <work>/runs/<run-name>/profile/{control,subject}/ (rocprofv3 CSV) and
# kernel-fraction reports; the queue log records PROFILE_EXIT like CAMPAIGN_EXIT.
set -u
arch=$1; dev=$2; run=$3; workload=$4; shift 4
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
: "${BC_HIP_PATH:?set BC_HIP_PATH}" "${BC_MODEL:?set BC_MODEL}"
toolchain=$(printf '%s' "$BC_HIP_PATH" | sha256sum | cut -c1-8)
builds="$work/builds/$arch-$toolchain"
log="$work/runs/$run.log"
ctl=$(grep -o 'control source: [^ ]*' "$log" | head -1 | awk -F/ '{print $NF}')
sub=$(grep -o 'subject source: [^ ]*' "$log" | head -1 | awk -F/ '{print $NF}')
bins=( "control:$builds/$ctl/control/bin/llama-bench" "subject:$builds/$sub/validation-subject/bin/llama-bench" )
case "$workload" in
  prefill) bench_args=(-p 512 -n 0 -r 3) ;;
  decode)  bench_args=(-p 0 -n 128 -r 3) ;;
  *) echo "unknown workload $workload"; echo "PROFILE_EXIT=2"; exit 2 ;;
esac
out="$work/runs/$run/profile"
mkdir -p "$out"
export TMPDIR=$work/tmp PYTHONPATH=tools ROCM_PATH=$BC_HIP_PATH HIP_PATH=$BC_HIP_PATH PATH=$BC_HIP_PATH/bin:$PATH
export HIP_VISIBLE_DEVICES=$dev
for kv in "$@"; do export "$kv"; done
status=0
for pair in "${bins[@]}"; do
  role=${pair%%:*}; bin=${pair#*:}
  if [ ! -x "$bin" ]; then echo "missing $role binary: $bin"; status=3; continue; fi
  rm -rf "$out/$role"; mkdir -p "$out/$role"
  rocprofv3 --kernel-trace --output-format csv -d "$out/$role" -o trace -- \
    "$bin" -m "$BC_MODEL" -ngl 99 "${bench_args[@]}" > "$out/$role/bench.log" 2>&1 || status=4
  csv=$(find "$out/$role" -name '*kernel_trace.csv' | head -1)
  if [ -z "$csv" ]; then echo "no kernel trace for $role"; status=5; continue; fi
  python3 -m bigcherry kernel-fraction --phase "$workload" --output "$out/$role/kernel-fraction.json" "$csv" \
    > "$out/$role/kernel-fraction.txt" 2>&1 || status=6
  echo "== $role"; head -25 "$out/$role/kernel-fraction.txt"
done
echo "PROFILE_EXIT=$status"
