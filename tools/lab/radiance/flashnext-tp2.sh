#!/bin/bash
# Qwen3.8-Flash-Next on two RX 7900 XTX through libr3: does radiance's shipped container (4-bit codebook experts,
# int8 trunk; 114 GiB) load and serve, and how fast.
# The container is written for two 32 GB RDNA4 cards (its card: the routed experts do not fit even those; the
# hottest stay in VRAM, the rest stream from a pinned host pool). On two 24 GB cards more of them stream, so the
# host pool is larger here and the placement is the tiered one. The first thing this run answers is whether libr3
# has a kernel for every op the container needs on gfx1100 - the engine names a missing one at startup.
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/flashnext-tp2.sh <tag>
# (The container's card also passes --expert-vs-cache-ratio 0.82; radiance 1.3.0 (89cee7c), the commit libr3's rewrite
# blocks are written against, does not have that option. Give it through MORE on a newer engine.)
# env: R (runs root), BUILD (b-main2), MODEL, CTX (32768), POOL_MIB (32768), HEADROOM_MIB (256), SPEC (0),
#      KV (fp8), WIRE (wht6), DEPTHS (2300,6500), MORE (further engine options), PROFILE=1 (--profile-ops)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
model=${MODEL:-/mnt/vault/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
extra="--tp-wire ${WIRE:-wht6} --placement expert_tiered --host-pool-mib ${POOL_MIB:-32768} --gpu-headroom-mib ${HEADROOM_MIB:-256} --max-num-batched-tokens 2048 ${MORE:-}"
[ "${PROFILE:-0}" = 1 ] && extra="$extra --profile-ops"
jobs=$(mktemp)
rm -rf "$R/$tag.log" "$R/$tag"
echo "VIS=${VIS:-0,1} SCRIPT $tag tools/lab/radiance/libr3-serve-smoke.sh @$b $R/$tag" > "$jobs"
env MODEL="$model" GPU=0,1 TP=2 CTX="${CTX:-32768}" KV="${KV:-fp8}" SPEC="${SPEC:-0}" LOAD_TIMEOUT=3000 BENCH=1 \
    BENCH_DEPTHS="${DEPTHS:-2300,6500}" BENCH_TOKENS=256 "EXTRA=$extra" \
    bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1 || {
        rm -f "$jobs"; echo "FLASHNEXT_QUEUE_FAILED $tag"; exit 1;
    }
rm -f "$jobs"
[ -s "$R/$tag.log" ] || { echo "FLASHNEXT_EMPTY_LOG $tag"; exit 1; }
if grep -qE 'SCRIPT_EXIT=[1-9][0-9]*|(^|[[:space:]])exit [1-9][0-9]*' "$R/$tag.log"; then
    echo "FLASHNEXT_JOB_FAILED $tag"; tail -10 "$R/$tag.log" | cut -c1-220; exit 1
fi
echo "== $tag [$extra]"
grep -iE "^prefill at|^decode, geometric|chunk geometry|exit [1-9]|^E |no kernel|refused|NO_|not load" "$R/$tag.log" | head -20 | cut -c1-200
grep -iE "^E|^F|no kernel|refused|unsupported|missing|settled at|resident|experts +[0-9]|host pool|file buffers" "$R/$tag/server.log" 2>/dev/null | head -40 | cut -c1-200
[ "${PROFILE:-0}" = 1 ] && python3 tools/lab/radiance/profile_ops_summary.py "$R/$tag/server.log" 22 2>&1 | head -30 | cut -c1-140
grep -qiE '^prefill at' "$R/$tag.log" || { echo "FLASHNEXT_NO_PREFILL_RESULT $tag"; exit 1; }
grep -qiE '^decode, geometric' "$R/$tag.log" || { echo "FLASHNEXT_NO_DECODE_RESULT $tag"; exit 1; }
echo "FLASHNEXT_DONE $tag"
