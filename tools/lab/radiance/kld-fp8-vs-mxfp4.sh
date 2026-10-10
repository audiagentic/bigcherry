#!/bin/bash
# RR06: KL divergence with the block-fp8 Qwen3.8-27B on the R9700 (libr4d) as the reference. No bf16 container of
# this model fits the cards, so the 8-bit one stands in for it; the numbers are distances from the 8-bit model, not
# from bf16.
#   fp8 on two XTX through libr3      the same weights on the other kernel library: what libr3's kernels add
#   MXFP4 on the R9700 (libr4d)       the 4-bit container against the 8-bit one
#   MXFP4 on two XTX through libr3    both differences together
# Start it directly (it queues its own jobs):
#   bash tools/lab/radiance/kld-fp8-vs-mxfp4.sh <tag>
# env: R (runs root), BUILD (b-main2), FP8, MXFP4 (containers), STEP_TOKENS (512)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
fp8=${FP8:-/mnt/data/llm-models/radiance/bigcherry/qwen3.8-27b-fp8/qwen3.8-27b-fp8-df2.rad}
mxfp4=${MXFP4:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
export REF=$R/$tag-ref/reference
# The KL mode keeps a logits row for every token of a step (2 GiB a rank at the default 4096), which the 29 GB
# container does not leave on the R9700 (run kld1); every run takes the same smaller step.
export EXTRA="--max-num-batched-tokens ${STEP_TOKENS:-512}"
jobs=$(mktemp)
one() {  # <vis> <run> <env...>
    local vis=$1 run=$2; shift 2
    rm -rf "$R/$run.log"; [ "$run" = "$tag-ref" ] || rm -rf "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/rad-kld.sh @$b $R/$run" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
    grep -vE "^radiance [0-9a-f]+;" "$R/$run.log" | grep -vE "SCRIPT_EXIT" | tail -16 | cut -c1-200
}
one 2 "$tag-ref" MODE=record MODEL="$fp8" GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref
one 0,1 "$tag-fp8-xtx" MODE=ref MODEL="$fp8" GPU=0,1 TP=2
one 2 "$tag-mxfp4-r4d" MODE=ref MODEL="$mxfp4" GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref
one 0,1 "$tag-mxfp4-xtx" MODE=ref MODEL="$mxfp4" GPU=0,1 TP=2
rm -f "$jobs"
echo "KLD_DONE $tag"
