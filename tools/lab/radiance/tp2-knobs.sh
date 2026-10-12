#!/bin/bash
# Two RX 7900 XTX, MXFP4 Qwen3.8-27B: which of the engine's own options move prefill, before any kernel work.
# Each arm is the serve bench (rad_prompt_bench.py: prefill at about 2.6K and 7.2K tokens, decode over 8 prompts),
# unprofiled, on the lossy wire unless the arm says otherwise:
#   base      --tp-wire wht6, prefill chunks of 2048 (the default checkpoint interval)
#   c4096     chunks of 4096 (--max-num-batched-tokens 4096 --checkpoint-interval 4096): half the all-reduces a token
#   c1024     chunks of 1024
#   minkb16   the lossy wire from 16 KiB instead of 128 KiB
#   kvfp8     fp8 KV cache instead of bf16
#   exact     the exact wire, for the size of the gap
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/tp2-knobs.sh <tag>
# env: R (runs root), BUILD (b-main2), ARMS (default all six), MODEL
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
arms=${ARMS:-base c4096 c1024 minkb16 kvfp8 exact}
common="GPU=0,1 TP=2 CTX=8192 BENCH=1 BENCH_DEPTHS=2300,6500 BENCH_TOKENS=256"
jobs=$(mktemp)
one() {  # <run> <env...>
    local run=$1; shift
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=0,1 SCRIPT $run tools/lab/radiance/libr3-serve-smoke.sh @$b $R/$run" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run"
    grep -iE "^prefill at|^decode, geometric|chunk geometry|exit [1-9]|^E " "$R/$run.log" | cut -c1-170
}
for a in $arms; do
    case $a in
        base)    one "$tag-base" $common "EXTRA=--tp-wire wht6" ;;
        c4096)   one "$tag-c4096" $common "EXTRA=--tp-wire wht6 --max-num-batched-tokens 4096 --checkpoint-interval 4096" ;;
        c1024)   one "$tag-c1024" $common "EXTRA=--tp-wire wht6 --max-num-batched-tokens 1024 --checkpoint-interval 1024" ;;
        minkb16) one "$tag-minkb16" $common "EXTRA=--tp-wire wht6 --tp-wire-min-kb 16" ;;
        kvfp8)   one "$tag-kvfp8" $common KV=fp8 "EXTRA=--tp-wire wht6" ;;
        exact)   one "$tag-exact" $common "EXTRA=--tp-wire exact" ;;
        *) echo "unknown arm $a" ;;
    esac
done
rm -f "$jobs"
echo "TP2_KNOBS_DONE $tag"
