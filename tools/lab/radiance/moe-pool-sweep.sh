#!/bin/bash
# What reading a routed expert from pinned host memory costs the w4 expert GEMM on one XTX: r3_selftest --perfmoe at
# decode (M=1) and prefill (M=2048) widths with 0, 50, 75 and 100% of the experts in the host pool. Says how much of
# Flash-Next's moe_gemm_q time on a 24 GB card is the link and how much is the kernel.
# libr3 must have been built first (libr3-build.sh).
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/moe-pool-sweep.sh <tag>
# env: R (runs root), BUILD (b-main2), MS ("1 2048"), POOLS ("0 50 75 100"), GPU (0), and libr3-perf.sh's own
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
jobs=$(mktemp)
for m in ${MS:-1 2048}; do
    for pool in ${POOLS:-0 50 75 100}; do
        run=$tag-m$m-p$pool
        rm -rf "$R/$run.log" "$R/$run"
        echo "VIS=${GPU:-0} SCRIPT $run tools/lab/radiance/libr3-perf.sh @$b $R/$run --perfmoe" > "$jobs"
        env GPU="${GPU:-0}" MOE_M="$m" MOE_POOL_PCT="$pool" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
        echo "######## M=$m pooled=$pool%"
        grep -q "^SCRIPT_EXIT=0" "$R/$run.log" || { echo "MOE_POOL_SWEEP_STOP $run failed"; tail -5 "$R/$run.log"; rm -f "$jobs"; exit 1; }
        grep -vE "^[0-9a-f:.]+ via |^radiance |^== |^SCRIPT_EXIT" "$R/$run.log" | cut -c1-200 | tail -14
    done
done
rm -f "$jobs"
echo "MOE_POOL_SWEEP_DONE $tag"
