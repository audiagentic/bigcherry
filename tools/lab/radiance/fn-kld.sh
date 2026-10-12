#!/bin/bash
# Flash-Next output quality on two XTX: does a libr3 switch (or the n-gram helper, or the stager) change what the
# model says? Flash-Next does not fit one card, so the reference is the same two-XTX engine with every native form off,
# the table on disk and the exact wire; each arm is scored against it (KLD, top-1 agreement, PPL ratio).
# The reference (tens of GiB) is recorded first and removed at the end unless KEEP_REF=1.
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/fn-kld.sh <tag>
# env: R, BUILD (b-main2), RADIANCE_SRC (engine tree, default the patched one), STEP_TOKENS (512), ARMS (see below)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
export RADIANCE_SRC=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance-patched}
export MODEL=${MODEL:-/mnt/vault/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
export REF=$R/$tag-ref/reference
w=/mnt/data/bigcherry-work/engines/radiance-rdna3/libr3-gfx1100-build/radiance_home/helpers/libr3_ngram_helper.so
a=/mnt/data/bigcherry-work/engines/radiance/build/radiance_home/kernels/libavx.so
base="--placement expert_tiered --host-pool-mib 24576 --gpu-headroom-mib 256 --kv-cache-dtype fp8 --max-num-batched-tokens ${STEP_TOKENS:-512}"
ng="R3_NGRAM_DEV=2 R3_NGRAM_HELPER_SO=$w R3_NGRAM_AVX=$a VISIBLE_GPU=0,1,2 KERNELS=libr3,libavx,libref"
jobs=$(mktemp)
one() {  # <vis> <run> <env...>
    local vis=$1 run=$2; shift 2
    rm -rf "$R/$run.log"; [ "$run" = "$tag-ref" ] || rm -rf "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/rad-kld.sh @$b $R/$run" > "$jobs"
    env GPU=0,1 TP=2 LOAD_TIMEOUT=3000 R3_ATTN_NATIVE=0 R3_GDN_NATIVE=0 R3_MOE_NATIVE=0 R3_HC_NATIVE=0 "$@" \
        bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
    grep -E "KLD mean|top-1 agreement|recorded:|exit [1-9]|^E " "$R/$run.log" | cut -c1-200
}
one 0,1 "$tag-ref" MODE=record KERNELS=libr3,libavx,libref "EXTRA=$base"
grep -q "recorded:" "$R/$tag-ref.log" || { echo "FN_KLD_STOP reference not recorded"; rm -f "$jobs"; exit 1; }
one 0,1,2 "$tag-ngram" MODE=ref $ng "EXTRA=$base"
one 0,1,2 "$tag-wire" MODE=ref $ng "EXTRA=$base --tp-wire wht6"
one 0,1,2 "$tag-native" MODE=ref $ng R3_ATTN_NATIVE=1 R3_MOE_NATIVE=1 R3_HC_NATIVE=1 "EXTRA=$base --tp-wire wht6"
rm -f "$jobs"
[ "${KEEP_REF:-0}" = 1 ] || rm -rf "$R/$tag-ref/reference"
echo "FN_KLD_DONE $tag"
