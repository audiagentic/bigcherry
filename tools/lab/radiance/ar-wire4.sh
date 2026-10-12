#!/bin/bash
# R3_AR_WIRE4 lab gate: build, two-XTX numeric selftest, three serve starts per arm, then KL quality.
# Usage: bash tools/lab/radiance/ar-wire4.sh <tag>
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
depth=${DEPTH:-6500}
mxfp4=${MXFP4:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
step="--max-num-batched-tokens ${STEP_TOKENS:-512}"
jobs=$(mktemp)
stop() { echo "AR_WIRE4_STOP $*"; rm -f "$jobs"; exit 1; }
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

one 0,1 "$tag-self" libr3-selftest.sh ar_rmsnorm_quant_fp8_host \
    GPU=0,1 R3_AR_WIRE4=0 R3_AR_PIPE=0 || stop selftest_queue
job_failed "$R/$tag-self.log" && stop selftest
grep -E "R3_AR_WIRE4|PASS|FAIL|verdict|error|Error" "$R/$tag-self.log" | tail -20 | cut -c1-240
grep -q "R3_AR_WIRE4 M=2048 dist=random" "$R/$tag-self.log" || stop selftest_random_missing
grep -q "R3_AR_WIRE4 M=2048 dist=heavy-tailed" "$R/$tag-self.log" || stop selftest_heavy_missing
grep -q "selftest verdict: PASS" "$R/$tag-self.log" || stop selftest

common="GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=$depth BENCH_TOKENS=4 N=4"
for arm in 0 1; do
    for i in 1 2 3; do
        run="$tag-wire4${arm}-$i"
        one 0,1 "$run" libr3-serve-smoke.sh - $common R3_AR_PIPE=0 R3_AR_WIRE4=$arm \
            "EXTRA=--tp-wire wht6" || stop "serve_queue $run"
        job_failed "$R/$run.log" && stop "serve $run"
        line=$(grep -iE "prefill.*tok/s|tok/s.*prefill|^prefill at" "$R/$run.log" | tail -1)
        [ -n "$line" ] || stop "no_prefill_rate $run"
        echo "AR_WIRE4_PREFILL arm=$arm start=$i $line" | cut -c1-240
    done
done

export MODEL=$mxfp4
export REF=$R/$tag-kld-ref/reference
one 2 "$tag-kld-ref" rad-kld.sh - MODE=record GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref \
    "EXTRA=$step" || stop kld_ref_queue
job_failed "$R/$tag-kld-ref.log" && stop kld_ref
grep -q "recorded:" "$R/$tag-kld-ref.log" || stop kld_ref_missing
[ -e "$REF" ] || stop kld_ref_path

for arm in 0 1; do
    run="$tag-kld-wire4$arm"
    one 0,1 "$run" rad-kld.sh - MODE=ref GPU=0,1 TP=2 R3_AR_PIPE=0 R3_AR_WIRE4=$arm \
        "EXTRA=$step --tp-wire wht6" || stop "kld_queue $run"
    job_failed "$R/$run.log" && stop "kld $run"
    kld=$(grep -E "KLD mean" "$R/$run.log" | tail -1)
    top=$(grep -E "top-1 agreement" "$R/$run.log" | tail -1)
    [ -n "$kld" ] || stop "kld_mean_missing $run"
    [ -n "$top" ] || stop "top1_missing $run"
    echo "AR_WIRE4_KLD arm=$arm $kld" | cut -c1-240
    echo "AR_WIRE4_KLD arm=$arm $top" | cut -c1-240
done

[ "${KEEP_REF:-0}" = 1 ] || rm -rf "$R/$tag-kld-ref/reference"
rm -f "$jobs"
echo "AR_WIRE4_DONE $tag"
