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
        bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run wire=$wire"
    grep -iE "^prefill at|exit [1-9]|^E " "$R/$run.log" | tail -4 | cut -c1-180
    python3 tools/lab/radiance/ar_phase_summary.py "$R/$run/rocprof" | cut -c1-240
}
for wire in $wires; do one "$tag-$wire" "$wire"; done
rm -f "$jobs"
echo "AR_PHASE_PROFILE_DONE $tag"
