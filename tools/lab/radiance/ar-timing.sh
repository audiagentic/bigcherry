#!/bin/bash
# R3 all-reduce phase timing on the two-XTX no-P2P path.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"

here=tools/lab/radiance
tag=${1:-}
[ -n "$tag" ] || { echo "usage: $0 <tag>" >&2; exit 2; }
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
b=${BUILD:-b-main2}
runs=${R:-/mnt/data/bigcherry-work/runs}
lab="$runs/$tag-ar-timing"
mkdir -p "$lab"

stop() {
    echo "AR_TIMING_STOP $*" >&2
    exit 1
}

"$here/libr3-rewrite-check.sh" > "$lab/rewrite.log" 2>&1 || {
    cat "$lab/rewrite.log" >&2
    stop rewrite
}
KBENCH=0 "$here/libr3-build.sh" /dev/null "$lab/build" > "$lab/build.log" 2>&1 || {
    grep -E "error:|Error|FAILED" "$lab/build.log" | head -30 >&2 || true
    stop build
}

build="$work/libr3-$target-build"
cmake --build "$build" --target r3_selftest > "$lab/selftest-build.log" 2>&1 || {
    grep -E "error:|Error|FAILED" "$lab/selftest-build.log" | head -30 >&2 || true
    stop selftest-build
}

for wire in wht6 exact; do
    run="$tag-$wire"
    out="$runs/$run"
    jobs=$(mktemp)
    rm -rf "$out.log" "$out"
    echo "VIS=0,1 SCRIPT $run $here/libr3-serve-smoke.sh @$b $out" > "$jobs"
    env R3_AR_TIMING=1 GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=6500 \
        BENCH_TOKENS=4 N=4 "EXTRA=--tp-wire $wire" \
        bash tools/lab/plan-qualification/queue.sh "$jobs" > "$lab/$wire.queue.log" 2>&1 || {
        cat "$lab/$wire.queue.log" >&2
        rm -f "$jobs"
        stop "$wire-serve"
    }
    rm -f "$jobs"
    [ -f "$out/server.log" ] || stop "$wire-no-server-log"
    lines=$(grep '^R3_AR_TIMING ' "$out/server.log" || true)
    [ -n "$lines" ] || stop "$wire-no-timing"
    printf '%s\n' "$lines"
done
