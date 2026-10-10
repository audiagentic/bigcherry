#!/bin/bash
# RR06: what part of a KL difference between the R9700 (libr4d) and the XTX cards (libr3) is the kernel library and
# what part is the two-card split. Run kld2 scored the block-fp8 27B on two XTX against the same container on the
# R9700 at KLD mean 0.0044 and 96.4% top-1 agreement, which is not zero and has two candidate causes.
# The MXFP4 container fits one XTX, so with it the three can be taken apart, each against MXFP4 on the R9700:
#   MXFP4 on the R9700 again          the floor: the same library and placement twice
#   MXFP4 on one XTX through libr3    the kernel library alone
#   MXFP4 on two XTX through libr3    the library and the split
# Start it directly (it queues its own jobs):
#   bash tools/lab/radiance/kld-libr3-control.sh <tag>
# env: R (runs root), BUILD (b-main2), MXFP4 (container), STEP_TOKENS (512)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
mxfp4=${MXFP4:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
export REF=$R/$tag-ref/reference
export EXTRA="--max-num-batched-tokens ${STEP_TOKENS:-512}"
export MODEL=$mxfp4
jobs=$(mktemp)
one() {  # <vis> <run> <env...>
    local vis=$1 run=$2; shift 2
    rm -rf "$R/$run.log"; [ "$run" = "$tag-ref" ] || rm -rf "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/rad-kld.sh @$b $R/$run" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
    grep -E "KLD mean|top-1 agreement|recorded:|exit [1-9]|^E " "$R/$run.log" | cut -c1-200
}
one 2 "$tag-ref" MODE=record GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref
one 2 "$tag-r4d-again" MODE=ref GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref
one 0 "$tag-xtx1" MODE=ref GPU=0
one 0,1 "$tag-xtx2" MODE=ref GPU=0,1 TP=2
rm -f "$jobs"
echo "KLD_CONTROL_DONE $tag"
