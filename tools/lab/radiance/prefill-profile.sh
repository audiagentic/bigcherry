#!/bin/bash
# Where a prefill's device time goes, op by op, on the cards being compared: one RX 7900 XTX and two (libr3, lossy
# wire) against the R9700 (libr4d). Prefill-heavy on purpose: one corpus prompt of about DEPTH tokens and a few
# generated tokens, so the per-op table is almost all prefill.
# The engine's per-op timing restores synchronisations, so speeds of these runs are not comparable with unprofiled
# ones; the shares and the per-call times are what to read (profile_ops_summary.py).
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/prefill-profile.sh <tag>
# env: R (runs root), BUILD (b-main2), MODEL (.rad container, default the MXFP4 27B), DEPTH (6500), CONFIGS
#      (default "xtx1 xtx2 r9700"), KV (bf16), EXTRA_ALL (more engine options for every config)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
configs=${CONFIGS:-xtx1 xtx2 r9700}
common="BENCH=1 BENCH_DEPTHS=${DEPTH:-6500} BENCH_TOKENS=4 N=4 CTX=8192"
jobs=$(mktemp)
one() {  # <vis> <run> <env...>
    local vis=$1 run=$2; shift 2
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/libr3-serve-smoke.sh @$b $R/$run" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "######## $run [$*]"
    grep -iE "^prefill at|exit [1-9]|^E " "$R/$run.log" | tail -3 | cut -c1-160
    python3 tools/lab/radiance/profile_ops_summary.py "$R/$run/server.log" 2>&1 | cut -c1-150
}
for c in $configs; do
    case $c in
        xtx1)  one 0 "$tag-xtx1" $common GPU=0 TP=1 "EXTRA=--profile-ops ${EXTRA_ALL:-}" ;;
        xtx2)  one 0,1 "$tag-xtx2" $common GPU=0,1 TP=2 "EXTRA=--profile-ops --tp-wire wht6 ${EXTRA_ALL:-}" ;;
        r9700) one 2 "$tag-r9700" $common GPU=2 TP=1 TARGET=gfx1201 KERNELS=libr4d,libref "EXTRA=--profile-ops ${EXTRA_ALL:-}" ;;
        *) echo "unknown config $c" ;;
    esac
done
rm -f "$jobs"
echo "PREFILL_PROFILE_DONE $tag"
