#!/bin/bash
# RR10: what the lossy all-reduce wire costs in output quality on two RX 7900 XTX.
# The two cards have no peer access, so every all-reduce goes through pinned host memory; radiance's --tp-wire wht6
# (a quantised payload for messages of 128 KiB and more) took two-card MXFP4 prefill from about 1,535 to 2,124 tok/s
# (runs r3-wht6*), with unchanged smoke replies and no quality number. This scores, against MXFP4 on the R9700:
#   two XTX, exact wire     the known distance of libr3 on two cards (0.0037 KLD, 96.8% top-1 in run kld3-xtx2)
#   two XTX, wht6           the same plus the lossy wire; the difference between the two rows is the wire
# The reference (about 20 GiB) is recorded first and removed at the end.
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/kld-tp-wire.sh <tag>
# env: R (runs root), BUILD (b-main2), MXFP4 (container), STEP_TOKENS (512), WIRE (wht6), KEEP_REF=1
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
mxfp4=${MXFP4:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
wire=${WIRE:-wht6}
step="--max-num-batched-tokens ${STEP_TOKENS:-512}"
export REF=$R/$tag-ref/reference
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
one 2 "$tag-ref" MODE=record GPU=2 TARGET=gfx1201 KERNELS=libr4d,libref "EXTRA=$step"
one 0,1 "$tag-xtx2-exact" MODE=ref GPU=0,1 TP=2 "EXTRA=$step"
one 0,1 "$tag-xtx2-$wire" MODE=ref GPU=0,1 TP=2 "EXTRA=$step --tp-wire $wire"
rm -f "$jobs"
[ "${KEEP_REF:-0}" = 1 ] || rm -rf "$R/$tag-ref/reference"
echo "KLD_TP_WIRE_DONE $tag"
