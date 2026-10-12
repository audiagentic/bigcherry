#!/bin/bash
# RR08: build libr3 with the lane-owns-a-row MXFP4 prefill GEMM and take it through selftest, row timing and the
# serve bench, each against the established tiled form on the same build (R3_BT=0 / 1).
# Stops after the selftest if the new form does not pass it. The long host-oracle matrix (R3_BT_ORACLE=1) is a
# separate step: ORACLE=1.
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/rr08-cycle.sh <tag>
# env: R (runs root), BUILD (b-main2), STEPS (default "build self perf serve1 serve2"), BT_B (8), ORACLE (0)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
steps=${STEPS:-build self perf serve1 serve2}
btb=${BT_B:-8}
shapes='34816 5120 64;34816 5120 128;34816 5120 512;5120 17408 64;5120 17408 128;5120 17408 512;16384 5120 64;16384 5120 128;16384 5120 512;5120 6144 64;5120 6144 128;5120 6144 512;12288 5120 64;12288 5120 128;12288 5120 512'
jobs=$(mktemp)
one() {  # <vis> <run> <script> <script args or -> <env...>
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
}
has() { case " $steps " in *" $1 "*) return 0;; esac; return 1; }

if has build; then
    one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0
    tail -4 "$R/$tag-build.log" | cut -c1-200
fi
if has self; then
    one 0 "$tag-self-old" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT=0 GPU=0
    grep -E "pass|fail|PASS|FAIL|error|exit" "$R/$tag-self-old.log" | tail -4 | cut -c1-200
    one 0 "$tag-self-bt" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT=1 R3_BT_B=$btb R3_BT_NT=1 GPU=0
    grep -E "pass|fail|PASS|FAIL|error|exit" "$R/$tag-self-bt.log" | tail -6 | cut -c1-200
    # "0 failed" is the pass line: stop only on a failed case or a FAIL verdict
    if grep -qE "[1-9][0-9]* failed|verdict: FAIL" "$R/$tag-self-bt.log" || ! grep -q "verdict: PASS" "$R/$tag-self-bt.log"; then echo "RR08_STOP selftest of the new form does not pass"; rm -f "$jobs"; exit 1; fi
fi
if [ "${ORACLE:-0}" = 1 ]; then
    one 0 "$tag-oracle-bt" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT_ORACLE=1 R3_BT=1 R3_BT_B=$btb R3_BT_NT=1 GPU=0 CASE_TIMEOUT=3600
    grep -E "pass|fail|PASS|FAIL|error|exit" "$R/$tag-oracle-bt.log" | tail -8 | cut -c1-200
fi
if has perf; then
    one 0 "$tag-perf-old" libr3-perf.sh --perfmxfp4 R3_BT=0 GPU=0 "SHAPES=$shapes" ITERS=20
    grep -vE "^$" "$R/$tag-perf-old.log" | tail -18 | cut -c1-170
    for bb in 4 6 8; do
        one 0 "$tag-perf-b$bb" libr3-perf.sh --perfmxfp4 R3_BT=1 R3_BT_B=$bb R3_BT_NT=1 GPU=0 "SHAPES=$shapes" ITERS=20
        grep -vE "^$" "$R/$tag-perf-b$bb.log" | tail -18 | cut -c1-170
    done
fi
bench="CTX=8192 BENCH=1 BENCH_DEPTHS=2300,6500 BENCH_TOKENS=256"
if has serve1; then
    one 0 "$tag-serve1-old" libr3-serve-smoke.sh - R3_BT=0 GPU=0 TP=1 $bench
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve1-old.log" | tail -8 | cut -c1-200
    one 0 "$tag-serve1-bt" libr3-serve-smoke.sh - R3_BT=1 R3_BT_B=$btb R3_BT_NT=1 GPU=0 TP=1 $bench
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve1-bt.log" | tail -8 | cut -c1-200
fi
if has serve2; then
    one 0,1 "$tag-serve2-old" libr3-serve-smoke.sh - R3_BT=0 GPU=0,1 TP=2 $bench "EXTRA=--tp-wire wht6"
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve2-old.log" | tail -8 | cut -c1-200
    one 0,1 "$tag-serve2-bt" libr3-serve-smoke.sh - R3_BT=1 R3_BT_B=$btb R3_BT_NT=1 GPU=0,1 TP=2 $bench "EXTRA=--tp-wire wht6"
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve2-bt.log" | tail -8 | cut -c1-200
fi
rm -f "$jobs"
echo "RR08_CYCLE_DONE $tag"
