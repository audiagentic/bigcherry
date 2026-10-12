#!/bin/bash
# RR08: build/select MXFP4 prefill forms and run selftest, row timing, ablations and serve benches.
# R3_BT=0 tiled; 1 direct one-wave; 2 eight-wave LDS-shared. Ablations are timing-only garbage.
# Stops after the selected candidate selftest if it fails.
# Usage: bash tools/lab/radiance/rr08-cycle.sh <tag>
# env: R, BUILD, STEPS, BT (2), BT_B (8), ABL (1,2,4,8), ORACLE (0), SWEEP_B (0)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
steps=${STEPS:-build self perf serve1 serve2}
bt=${BT:-2}
btb=${BT_B:-8}
abl=${ABL:-1,2,4,8}
shapes='34816 5120 64;34816 5120 128;34816 5120 512;5120 17408 64;5120 17408 128;5120 17408 512;16384 5120 64;16384 5120 128;16384 5120 512;5120 6144 64;5120 6144 128;5120 6144 512;12288 5120 64;12288 5120 128;12288 5120 512'
jobs=$(mktemp)
one() {
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
}
has() { case " $steps " in *" $1 "*) return 0;; esac; return 1; }
show_self() { grep -E "pass|fail|PASS|FAIL|error|exit" "$1" | tail -6 | cut -c1-200; }
failed_self() { grep -qE "[1-9][0-9]* failed|verdict: FAIL" "$1" || ! grep -q "verdict: PASS" "$1"; }

if has build; then
    one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0
    tail -4 "$R/$tag-build.log" | cut -c1-200
fi
if has self; then
    one 0 "$tag-self-old" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT=0 R3_BT_ABL=0 GPU=0
    show_self "$R/$tag-self-old.log"
    one 0 "$tag-self-bt$bt" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT=$bt R3_BT_B=$btb R3_BT_NT=1 R3_BT_ABL=0 GPU=0
    show_self "$R/$tag-self-bt$bt.log"
    if failed_self "$R/$tag-self-bt$bt.log"; then
        echo "RR08_STOP selftest of R3_BT=$bt does not pass"; rm -f "$jobs"; exit 1
    fi
fi
if [ "${ORACLE:-0}" = 1 ]; then
    one 0 "$tag-oracle-bt$bt" libr3-selftest.sh gemm_mxfp4a8_tiled R3_BT_ORACLE=1 R3_BT=$bt R3_BT_B=$btb R3_BT_NT=1 R3_BT_ABL=0 GPU=0 CASE_TIMEOUT=3600
    show_self "$R/$tag-oracle-bt$bt.log"
fi
if has abl; then
    echo "RR08_ABL timing only; outputs are intentionally garbage; bits: 1=weight 2=activation 4=wmma 8=epilogue"
    for mask in ${abl//,/ }; do
        for mode in 0 "$bt"; do
            one 0 "$tag-abl-bt$mode-m$mask" libr3-perf.sh --perfmxfp4 R3_BT=$mode R3_BT_B=$btb R3_BT_NT=1 R3_BT_ABL=$mask GPU=0 "SHAPES=$shapes" ITERS=20
            grep -vE "^$" "$R/$tag-abl-bt$mode-m$mask.log" | tail -18 | cut -c1-170
        done
    done
fi
if has perf; then
    one 0 "$tag-perf-old" libr3-perf.sh --perfmxfp4 R3_BT=0 R3_BT_ABL=0 GPU=0 "SHAPES=$shapes" ITERS=20
    grep -vE "^$" "$R/$tag-perf-old.log" | tail -18 | cut -c1-170
    if [ "${SWEEP_B:-0}" = 1 ]; then bs="4 6 8"; else bs="$btb"; fi
    for bb in $bs; do
        one 0 "$tag-perf-bt$bt-b$bb" libr3-perf.sh --perfmxfp4 R3_BT=$bt R3_BT_B=$bb R3_BT_NT=1 R3_BT_ABL=0 GPU=0 "SHAPES=$shapes" ITERS=20
        grep -vE "^$" "$R/$tag-perf-bt$bt-b$bb.log" | tail -18 | cut -c1-170
    done
fi
bench="CTX=8192 BENCH=1 BENCH_DEPTHS=2300,6500 BENCH_TOKENS=256"
if has serve1; then
    one 0 "$tag-serve1-old" libr3-serve-smoke.sh - R3_BT=0 R3_BT_ABL=0 GPU=0 TP=1 $bench
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve1-old.log" | tail -8 | cut -c1-200
    one 0 "$tag-serve1-bt$bt" libr3-serve-smoke.sh - R3_BT=$bt R3_BT_B=$btb R3_BT_NT=1 R3_BT_ABL=0 GPU=0 TP=1 $bench
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve1-bt$bt.log" | tail -8 | cut -c1-200
fi
if has serve2; then
    one 0,1 "$tag-serve2-old" libr3-serve-smoke.sh - R3_BT=0 R3_BT_ABL=0 GPU=0,1 TP=2 $bench "EXTRA=--tp-wire wht6"
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve2-old.log" | tail -8 | cut -c1-200
    one 0,1 "$tag-serve2-bt$bt" libr3-serve-smoke.sh - R3_BT=$bt R3_BT_B=$btb R3_BT_NT=1 R3_BT_ABL=0 GPU=0,1 TP=2 $bench "EXTRA=--tp-wire wht6"
    grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$R/$tag-serve2-bt$bt.log" | tail -8 | cut -c1-200
fi
rm -f "$jobs"
echo "RR08_CYCLE_DONE $tag"
