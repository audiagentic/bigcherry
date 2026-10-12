#!/bin/bash
# RR09-A: measure the gfx1100 matrix arithmetic roof before any more MXFP4 prefill-GEMM work.
# Queues its own one-card jobs through one(), matching rr08-cycle.sh.
# Usage: bash tools/lab/radiance/wmma-roof.sh <tag>
# env: R, BUILD, ITERS (8192), STEPS ('build perf')
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
steps=${STEPS:-build perf}
jobs=$(mktemp)
one() {
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    if ! env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1; then
        echo "WMMA_ROOF_QUEUE_FAILED $run"; return 1
    fi
    [ -s "$R/$run.log" ] || { echo "WMMA_ROOF_EMPTY_LOG $run"; return 1; }
    if grep -qE 'SCRIPT_EXIT=[1-9][0-9]*|(^|[[:space:]])exit [1-9][0-9]*' "$R/$run.log"; then
        echo "WMMA_ROOF_JOB_FAILED $run"; tail -8 "$R/$run.log" | cut -c1-220; return 1
    fi
    echo "== $run [$*]"
}
has() { case " $steps " in *" $1 "*) return 0;; esac; return 1; }

if has build; then
    one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0 || { rm -f "$jobs"; exit 1; }
    tail -5 "$R/$tag-build.log" | cut -c1-220
fi
if has perf; then
    one 0 "$tag-wmma" libr3-perf.sh --perfwmma GPU=0 ITERS=${ITERS:-8192} || { rm -f "$jobs"; exit 1; }
    grep -E "^WMMA|^BEST|SKIP|FAILED|failed|error" "$R/$tag-wmma.log" | cut -c1-240
    grep -qE '^WMMA|^BEST' "$R/$tag-wmma.log" || { echo "WMMA_ROOF_EMPTY_RESULTS $tag-wmma"; rm -f "$jobs"; exit 1; }
fi
rm -f "$jobs"
echo "WMMA_ROOF_DONE $tag"
