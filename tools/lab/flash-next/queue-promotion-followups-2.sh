#!/bin/bash
# Second set of follow-up runs for the 2026-10-09 promotion candidates (owner decisions of that day), on the lab tree:
#   5. 1356 stress: STRESS_N (26) ABBAs at 8K on b-metamem-mdw3 = 52 threaded runs; every text must equal the baseline.
#   6. 1355 more rounds on Flash-Next: 4 ABBAs at 8K and at 24K with a 2048-token decode, then 2 ABBAs at 24K with a
#      second request text. A = fused (default), B = BIGCHERRY_HC_POST_GATE_FUSE=0.
#   7. Other models, tensor split, no draft (probe-27b.sh plain), ABBA with the patch's off switch:
#      Gemma 4 26B A4B and Qwen3.6-35B-A3B, for 1358 (b-metamem-sce1) and 1355 (b-metamem-hcf1).
#      1355 only acts on Qwen4Exp hyper-connection gates, so on these models it can only show "no change".
#   8. 1357 router split-K on the same two models (b-metamem-rse1), off against on.
# Waits for queue-promotion-followups.sh (FOLLOWUPS_DONE in WAIT_LOG) when WAIT_LOG is set.
# Usage: queue-promotion-followups-2.sh [step...]     (default: 5 6 7)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
FLASH=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
GEMMA=/mnt/data/llm-models/gemma-4-26B-A4B/gguf/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf
Q36=/mnt/data/llm-models/qwen3.6-35B-A3B/gguf/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
steps="${*:-5 6 7}"
[ -n "${WAIT_LOG:-}" ] && until grep -q FOLLOWUPS_DONE "$WAIT_LOG"; do sleep 60; done

texts() {  # <glob of run dirs> <arm> -> "md5prefix xN ..."
    md5sum $1/*-$2/*.greedy.txt 2> /dev/null | awk '{print $1}' | sort | uniq -c | awk '{printf "%s x%s ", substr($2, 1, 8), $1}'
}

other_model() {  # <label> <model> <build> <off-switch env> <tag>
    local label=$1 model=$2 run=$3 off=$4 tag=$5 i=0 arm e j
    for arm in A B B A; do
        i=$((i + 1)); j=$(mktemp)
        e=""; [ $arm = B ] && e="$off"
        echo "VIS=0,1,2,3 SCRIPT $tag-$i-$arm tools/lab/dflash/probe-27b.sh @$run $R/$tag-$i-$arm (plain) 3" > "$j"
        env PROBE_MODEL=$model BC_MODEL=$model BIGCHERRY_PATCH_TRACE=1 $e bash tools/lab/plan-qualification/queue.sh "$j" > "$R/$tag-$i-$arm.queue.log" 2>&1
        rm -f "$j"
        echo "-- $label $i $arm ${e:-default}: $(grep -rhE "^plain: |SERVER_FAILED" $R/$tag-$i-$arm.log $R/$tag-$i-$arm/ 2> /dev/null | sort -u | head -2 | tr '\n' ' ') md5 $(md5sum $R/$tag-$i-$arm/*.greedy.txt 2> /dev/null | cut -c1-12 | sort -u | tr '\n' ' ') hits $(grep -rhc "BIGCHERRY_PATCH_HIT patch=13[5][58]" $R/$tag-$i-$arm/ 2> /dev/null | paste -sd+ | bc 2> /dev/null)"
    done
}

for s in $steps; do
    case "$s" in
    5)
        echo "== step 5: 1356 stress, ${STRESS_N:-26} ABBAs at 8K"
        export BC_MODEL=$FLASH
        rm -rf $R/mdw3-s[0-9]*
        for r in $(seq ${STRESS_N:-26}); do
            AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" bash tools/lab/flash-next/queue-env-ab.sh mdw3-s$r b-metamem-mdw3 8192 | grep -E "^md5|SERVER_FAILED|blocked" | sed "s/^/s$r /" | cut -c1-80
        done
        echo "stress texts: A (baseline) $(texts "$R/mdw3-s*-d8192" A); B (threads) $(texts "$R/mdw3-s*-d8192" B)"
        echo "server failures: $(grep -l SERVER_FAILED $R/mdw3-s*-d8192.log 2> /dev/null | wc -l)"
        ;;
    6)
        echo "== step 6: 1355 more rounds on Flash-Next (A = fused, B = BIGCHERRY_HC_POST_GATE_FUSE=0)"
        export BC_MODEL=$FLASH
        rm -rf $R/hcf1-m*
        for r in 1 2 3 4; do
            DECODE_N=2048 AB_ENV="BIGCHERRY_HC_POST_GATE_FUSE=0" bash tools/lab/flash-next/queue-env-ab.sh hcf1-m$r b-metamem-hcf1 8192 24576 | sed "s/^d/m$r d/" | grep -E "^==|^m[0-9]|^md5|SERVER_FAILED|blocked" | cut -c1-150
        done
        export ASK="Write the next chapter of the text above, in the same style:"
        for r in 5 6; do
            DECODE_N=2048 AB_ENV="BIGCHERRY_HC_POST_GATE_FUSE=0" bash tools/lab/flash-next/queue-env-ab.sh hcf1-m$r b-metamem-hcf1 24576 | sed "s/^d/m$r(ask) d/" | grep -E "^==|^m[0-9]|^md5|SERVER_FAILED|blocked" | cut -c1-150
        done
        unset ASK
        ;;
    7)
        echo "== step 7: other models, tensor split, no draft"
        rm -rf $R/om-*
        other_model "gemma 1358" $GEMMA b-metamem-sce1 BIGCHERRY_META_SPLIT_CACHE_EVICT=0 om-gem-1358
        other_model "qwen3.6 1358" $Q36 b-metamem-sce1 BIGCHERRY_META_SPLIT_CACHE_EVICT=0 om-q36-1358
        other_model "gemma 1355" $GEMMA b-metamem-hcf1 BIGCHERRY_HC_POST_GATE_FUSE=0 om-gem-1355
        other_model "qwen3.6 1355" $Q36 b-metamem-hcf1 BIGCHERRY_HC_POST_GATE_FUSE=0 om-q36-1355
        ;;
    8)
        # 1357 is off by default, so here B = the router kernel ON (BIGCHERRY_MOE_ROUTER_SPLITK=1), A = off.
        # Qwen3.6-35B-A3B is MoE and builds its FFN through build_moe_ffn, so the router is marked there too;
        # Gemma 4 has no router and can only show "no change, 0 hits".
        echo "== step 8: 1357 router split-K on other models (A = off, B = on)"
        rm -rf $R/om-*-1357*
        other_model "qwen3.6 1357" $Q36 b-metamem-rse1 BIGCHERRY_MOE_ROUTER_SPLITK=1 om-q36-1357
        other_model "gemma 1357" $GEMMA b-metamem-rse1 BIGCHERRY_MOE_ROUTER_SPLITK=1 om-gem-1357
        ;;
    *) echo "unknown step $s" ;;
    esac
done
echo FOLLOWUPS2_DONE
