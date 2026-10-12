#!/bin/bash
# Flash-Next native-MoE lab cycle on two gfx1100 cards.
# Build -> bit comparison at decode/prefill -> --perfmoe old/new -> full TP2 serve old/new.
# Usage: bash tools/lab/radiance/fn-moe-cycle.sh <tag>
# env: R, BUILD, STEPS ("build self perf serve"), MOE_ITERS (10), plus flashnext-tp2.sh model/config env.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
steps=${STEPS:-build self perf serve}
jobs=$(mktemp)
trap 'rm -f "$jobs"' EXIT
has() { case " $steps " in *" $1 "*) return 0;; esac; return 1; }
one() {
    local run=$1 script=$2 args=$3; shift 3
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=0 SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    if ! env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1; then
        echo "FN_MOE_QUEUE_FAILED $run"; return 1
    fi
    [ -s "$R/$run.log" ] || { echo "FN_MOE_EMPTY_LOG $run"; return 1; }
    if grep -qE 'SCRIPT_EXIT=[1-9][0-9]*|(^|[[:space:]])exit [1-9][0-9]*' "$R/$run.log"; then
        echo "FN_MOE_JOB_FAILED $run"; tail -10 "$R/$run.log" | cut -c1-220; return 1
    fi
    echo "== $run [$*]"
}
if has build; then
    one "$tag-build" libr3-build.sh - KBENCH=0 GPU=0 || { echo "FN_MOE_BUILD_FAILED"; exit 1; }
    grep -E 'R3_REWRITE_CHECK_|build exit|CONFIGURE_FAILED|FAIL' "$R/$tag-build.log" | tail -12 | cut -c1-220
fi
if has self; then
    for m in 1 2048; do
        run="$tag-self-m$m"
        one "$run" libr3-perf.sh --perfmoe GPU=0 MOE_M=$m ITERS=1 R3_MOE_COMPARE=1 R3_MOE_NATIVE=0 || {
            echo "FN_MOE_SELFTEST_JOB_FAILED M=$m"; exit 1;
        }
        grep -E 'R3_MOE_COMPARE_|g64.*form' "$R/$run.log" | cut -c1-220
        [ "$(grep -c '^R3_MOE_COMPARE_PASS ' "$R/$run.log")" -ge 2 ] || {
            echo "FN_MOE_SELFTEST_FAILED M=$m"; exit 1;
        }
        ! grep -q '^R3_MOE_COMPARE_FAIL ' "$R/$run.log" || { echo "FN_MOE_SELFTEST_FAILED M=$m"; exit 1; }
    done
fi
if has perf; then
    for m in 1 2048; do
        for sw in 0 1; do
            run="$tag-perf-m$m-r3$sw"
            one "$run" libr3-perf.sh --perfmoe GPU=0 MOE_M=$m ITERS=${MOE_ITERS:-10} R3_MOE_COMPARE=0 R3_MOE_NATIVE=$sw || {
                echo "FN_MOE_PERF_FAILED M=$m R3_MOE_NATIVE=$sw"; exit 1;
            }
            grep -E 'g64.*form|PERFMOE_FAILED' "$R/$run.log" | cut -c1-220
        done
    done
fi
if has serve; then
    for sw in 0 1; do
        run="$tag-serve-r3$sw"
        if ! R3_MOE_NATIVE=$sw bash tools/lab/radiance/flashnext-tp2.sh "$run"; then
            echo "FN_MOE_SERVE_FAILED R3_MOE_NATIVE=$sw"; exit 1
        fi
    done
fi
echo "FN_MOE_CYCLE_DONE $tag"
