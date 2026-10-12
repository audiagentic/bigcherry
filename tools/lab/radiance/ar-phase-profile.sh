#!/bin/bash
# RR10-B: phase trace the no-P2P two-XTX all-reduce on exact and wht6 wires.
# Usage: bash tools/lab/radiance/ar-phase-profile.sh <tag>
# env: R, BUILD, WIRES ("exact wht6"), DEPTH (6500), MODEL
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
wires=${WIRES:-exact wht6}
jobs=$(mktemp)
one() {
    local run=$1 wire=$2
    rm -rf "$R/$run.log" "$R/$run"
    mkdir -p "$R/$run"
    echo "VIS=0,1 SCRIPT $run tools/lab/radiance/libr3-serve-smoke.sh @$b $R/$run" > "$jobs"
    env GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=${DEPTH:-6500} BENCH_TOKENS=4 N=4 \
        "EXTRA=--tp-wire $wire" \
        RADIANCE_PREFIX="$PWD/tools/lab/radiance/rocprof-runtime.sh" \
        ROCPROF_OUT="$R/$run/rocprof" \
        bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1 || {
            echo "AR_PHASE_PROFILE_QUEUE_FAILED $run"; return 1;
        }
    [ -s "$R/$run.log" ] || { echo "AR_PHASE_PROFILE_EMPTY_LOG $run"; return 1; }
    if grep -qE 'SCRIPT_EXIT=[1-9][0-9]*|(^|[[:space:]])exit [1-9][0-9]*' "$R/$run.log"; then
        echo "AR_PHASE_PROFILE_JOB_FAILED $run"; tail -8 "$R/$run.log" | cut -c1-220; return 1
    fi
    trace="$R/$run/rocprof"
    [ -d "$trace" ] && [ -n "$(find "$trace" -type f -name '*_trace.csv' -size +0c -print -quit 2>/dev/null)" ] || {
        echo "AR_PHASE_PROFILE_EMPTY_TRACE $run"; return 1;
    }
    echo "== $run wire=$wire"
    grep -iE "^prefill at|exit [1-9]|^E " "$R/$run.log" | tail -4 | cut -c1-180
    python3 tools/lab/radiance/ar_phase_summary.py "$trace" > "$R/$run/ar-phase-summary.txt" || {
        echo "AR_PHASE_PROFILE_SUMMARY_FAILED $run"; return 1;
    }
    cut -c1-240 "$R/$run/ar-phase-summary.txt"
    grep -qE '^AR/quant kernels[[:space:]]+calls=[[:space:]]*[1-9][0-9]*' "$R/$run/ar-phase-summary.txt" || {
        echo "AR_PHASE_PROFILE_EMPTY_EVENTS $run"; return 1;
    }
}
for wire in $wires; do
    one "$tag-$wire" "$wire" || { rm -f "$jobs"; exit 1; }
done
rm -f "$jobs"
echo "AR_PHASE_PROFILE_DONE $tag"
