#!/bin/bash
# Cross-model check of the bc-11474.1.0 production set (first tested on Flash-Next only): 1348 deferred MTP catch-up
# (default on) and 1321/1322 MTP look-ahead (BIGCHERRY_MTP_AHEAD, on in the flashnext profile only).
#  1. Qwen3.8-27B Q8_0, dual-XTX production flags, built-in MTP (prod27b-ab.sh, 10K + 32K, same binary on both arms):
#     B = deferred catch-up off; then B = look-ahead on. Two ABBA repeats each.
#  2. Gemma 4 26B A4B UD-Q5_K_S, tensor split, no draft (probe-27b.sh plain): default vs deferred catch-up off, ABBA.
#     No MTP, so this is identity + no-regression only.
#  3. Flash-Next with a different request after the fill (ASK): deferred off, and look-ahead off, at 24K.
# Activation is reported per model: a patch that never fires on a model proves nothing about it there.
# Usage: cross-model-rel.sh <tag>   (run from the lab tree on main; builds the production set as b-metamem-<tag>)
set -u
TAG=${1:?tag}
R=/mnt/data/bigcherry-work/runs
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
FN=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
GEMMA=/mnt/data/llm-models/gemma-4-26B-A4B/gguf/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf
RUN=b-metamem-$TAG
export BC_MODEL=$FN   # queue.sh requires it; each job sets its own model

echo "== build + Flash-Next smoke ($(git log --oneline -1 | cut -c1-60))"
BC_MODEL=$FN EXPERIMENT=stock-none CTX_LIST=245760 DEPTH=8192 bash tools/lab/flash-next/queue-meta-mem.sh "$TAG" | grep -E "^timing|SERVER_FAILED|error lines|QUEUE_EXIT"

hits() {  # <dir>: which of today's mechanisms fired in the server logs under it
    for p in 1348_mtp_deferred 1322_mtp_ahead 1321_mtp_ahead; do
        echo "   PATCH_HIT $p: $(grep -rhc "PATCH_HIT patch=$p" "$1" 2>/dev/null | paste -sd+ | bc 2>/dev/null)"
    done
}

echo "== 1. Qwen3.8-27B dual XTX, built-in MTP"
export AR_FLAG="--allreduce cpu-root"
for spec in "defer:BIGCHERRY_MTP_DEFERRED_CATCHUP=0" "ahead:BIGCHERRY_MTP_AHEAD=1"; do
    name=${spec%%:*}; envs=${spec#*:}
    for r in 1 2; do
        j=$(mktemp)
        echo "VIS=0,1 SCRIPT x27-$TAG-$name-r$r tools/lab/flash-next/prod27b-ab.sh @$RUN @$RUN $R/x27-$TAG-$name-r$r BIGCHERRY_PATCH_TRACE=1 $envs" > "$j"
        bash tools/lab/plan-qualification/queue.sh "$j" > "$R/x27-$TAG-$name-r$r.queue.log" 2>&1; rm -f "$j"
        echo "-- 27B A = default, B = $envs (repeat $r)"
        grep -E "^d[0-9]|SERVER_FAILED" "$R/x27-$TAG-$name-r$r.log"
        for d in 10240 32768; do
            echo "   md5 d$d A: $(md5sum $R/x27-$TAG-$name-r$r/d$d.*.A.txt 2>/dev/null | cut -c1-12 | sort -u | tr '\n' ' ') B: $(md5sum $R/x27-$TAG-$name-r$r/d$d.*.B.txt 2>/dev/null | cut -c1-12 | sort -u | tr '\n' ' ')"
        done
    done
    hits "$R/x27-$TAG-$name-r1"
done
unset AR_FLAG

echo "== 2. Gemma 4 26B A4B, no draft"
i=0
for arm in A B B A; do
    i=$((i + 1)); j=$(mktemp)
    e=""; [ $arm = B ] && e="BIGCHERRY_MTP_DEFERRED_CATCHUP=0"
    echo "VIS=0,1,2,3 SCRIPT xgem-$TAG-$i-$arm tools/lab/dflash/probe-27b.sh @$RUN $R/xgem-$TAG-$i-$arm (plain) 3" > "$j"
    env PROBE_MODEL=$GEMMA BC_MODEL=$GEMMA $e bash tools/lab/plan-qualification/queue.sh "$j" > "$R/xgem-$TAG-$i-$arm.queue.log" 2>&1; rm -f "$j"
    echo "-- gemma $i $arm ${e:-default}: $(grep -rhE "^plain: |SERVER_FAILED" $R/xgem-$TAG-$i-$arm.log $R/xgem-$TAG-$i-$arm/ 2>/dev/null | sort -u | head -2 | tr '\n' ' ') md5 $(md5sum $R/xgem-$TAG-$i-$arm/*.greedy.txt 2>/dev/null | cut -c1-12 | sort -u | tr '\n' ' ')"
done

echo "== 3. Flash-Next, second request text"
export ASK="Write the next chapter of the text above, in the same style:"
BC_MODEL=$FN AB_ENV="BIGCHERRY_MTP_DEFERRED_CATCHUP=0" bash tools/lab/flash-next/queue-env-ab.sh xfn-$TAG-defer $RUN 24576 | grep -E "^==|^d[0-9]|^md5|SERVER_FAILED|blocked"
BC_MODEL=$FN AB_ENV="BIGCHERRY_MTP_AHEAD=0" bash tools/lab/flash-next/queue-env-ab.sh xfn-$TAG-ahead $RUN 24576 | grep -E "^==|^d[0-9]|^md5|SERVER_FAILED|blocked"
echo CROSS_MODEL_DONE
