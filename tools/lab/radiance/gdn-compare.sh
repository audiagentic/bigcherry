#!/bin/bash
# Build libr3, then run the served 27B MXFP4 TP2 GDN native/compat comparator.
# Usage: bash tools/lab/radiance/gdn-compare.sh <tag>
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
jobs=$(mktemp)
stop() { echo "GDN_COMPARE_STOP $*"; rm -f "$jobs"; exit 1; }
one() {
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1 || return 1
}
script_exit() {
    grep -oE 'SCRIPT_EXIT=[0-9]+' "$1" | tail -1 | cut -d= -f2
}

one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0 || stop build_queue
be=$(script_exit "$R/$tag-build.log")
[ "$be" = 0 ] || stop "build SCRIPT_EXIT=${be:-missing}"
grep -E "R3_REWRITE_CHECK|build exit|CONFIGURE_FAILED|error:|Error" "$R/$tag-build.log" | tail -12 | cut -c1-240

one 0,1 "$tag-serve" libr3-serve-smoke.sh - \
    GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=2300 BENCH_TOKENS=4 N=2 \
    R3_GDN_NATIVE=1 R3_GDN_COMPARE=1 || stop serve_queue
se=$(script_exit "$R/$tag-serve.log")
[ "$se" = 0 ] || stop "serve SCRIPT_EXIT=${se:-missing}"
grep -E "R3_GDN_COMPARE (mismatch|setup_failed)" "$R/$tag-serve/server.log" "$R/$tag-serve.log" 2>/dev/null || \
    echo "R3_GDN_COMPARE no mismatch"
rm -f "$jobs"
echo "GDN_COMPARE_DONE $tag"
