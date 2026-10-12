#!/bin/bash
# R3_AR_PIPE lab gate: build, two-XTX byte selftest, then three baseline and three pipelined starts.
# Usage: bash tools/lab/radiance/ar-pipe.sh <tag>
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
depth=${DEPTH:-6500}
jobs=$(mktemp)
stop() { echo "AR_PIPE_STOP $*"; rm -f "$jobs"; exit 1; }
one() {
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1 || return 1
}
job_failed() { grep -qE "SCRIPT_EXIT=[1-9][0-9]*|(^|[[:space:]])exit [1-9][0-9]*" "$1"; }

one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0 || stop build_queue
job_failed "$R/$tag-build.log" && stop build
grep -E "libr3 build|REWRITE|error|Error" "$R/$tag-build.log" | tail -6 | cut -c1-220

one 0,1 "$tag-self" libr3-selftest.sh ar_rmsnorm_quant_fp8_host GPU=0,1 || stop selftest_queue
grep -E "R3_AR_PIPE|PASS|FAIL|verdict|error|Error" "$R/$tag-self.log" | tail -12 | cut -c1-220
grep -q "selftest verdict: PASS" "$R/$tag-self.log" || stop selftest
job_failed "$R/$tag-self.log" && stop selftest

common="GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=$depth BENCH_TOKENS=4 N=4"
for arm in 0 1; do
    for i in 1 2 3; do
        run="$tag-pipe${arm}-$i"
        one 0,1 "$run" libr3-serve-smoke.sh - $common R3_AR_PIPE=$arm "EXTRA=--tp-wire wht6" || stop "serve_queue $run"
        job_failed "$R/$run.log" && stop "serve $run"
        line=$(grep -iE "prefill.*tok/s|tok/s.*prefill|^prefill at" "$R/$run.log" | tail -1)
        [ -n "$line" ] || stop "no_prefill_rate $run"
        echo "AR_PIPE_PREFILL arm=$arm start=$i $line" | cut -c1-240
    done
done
rm -f "$jobs"
echo "AR_PIPE_DONE $tag"
