#!/bin/bash
# Follow-up runs for the 2026-10-09 promotion candidates, one after another on the lab tree (main checked out):
#   1. 1357 router split-K: accuracy probes against the CPU f32 reference at 8K (build b-metamem-rsk1).
#   2. 1356 dispatch workers: rebuild after the anchor fix (b-metamem-mdw3), ABBA at 8K and 24K twice, marker.
#   3. Qwen3.8-27B dual-XTX no-regression (prod27b-ab.sh): production b-main2 against the 1355, 1358 and 1356 builds.
#   4. QFP42 step 0: blocked-time trace, deferred catch-up on against off, at 24K and 98K (1346 and 1320 timing lines).
# Usage: queue-promotion-followups.sh [step...]     (default: 1 2 3 4)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
FLASH=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
steps="${*:-1 2 3 4}"

queue_one() {  # <job line>
    local j
    j=$(mktemp)
    echo "$1" > "$j"
    bash tools/lab/plan-qualification/queue.sh "$j" | grep -E "rc=|blocked"
    rm -f "$j"
}

for s in $steps; do
    case "$s" in
    1)
        echo "== step 1: 1357 accuracy probes at 8K"
        export BC_MODEL=$FLASH
        rm -rf $R/rsk1-fid $R/rsk1-fid.log
        queue_one "VIS=0,1,2,3 SCRIPT rsk1-fid tools/lab/flash-next/flash-fidelity.sh 8192 @b-metamem-rsk1 $R/rsk1-fid ref BIGCHERRY_MOE_ROUTER_SPLITK=1"
        grep -vE "PCIe" $R/rsk1-fid.log | cut -c1-260
        ;;
    2)
        echo "== step 2: 1356 after the anchor fix"
        export BC_MODEL=$FLASH BIGCHERRY_PATCH_TRACE=1
        rm -rf $R/b-metamem-mdw3.log $R/metamem-mdw3* $R/mdw3-*
        EXPERIMENT=meta-dispatch-workers CTX_LIST=245760 DEPTH=8192 bash tools/lab/flash-next/queue-meta-mem.sh mdw3 | grep -E "^timing|SERVER_FAILED|QUEUE_EXIT"
        grep -hE "^BUILD_(EXIT|BINARY)" $R/b-metamem-mdw3.log | cut -c1-160
        grep -hE "error:" $R/b-metamem-mdw3.log | sort -u | head -8 | cut -c1-220
        for r in 1 2; do
            AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" bash tools/lab/flash-next/queue-env-ab.sh mdw3-r$r b-metamem-mdw3 8192 24576 | sed "s/^d/r$r d/" | grep -E "^==|^r[0-9]|^md5|SERVER_FAILED|blocked" | cut -c1-150
        done
        echo "marker (A B B A at 24K): $(for d in $R/mdw3-r1-d24576/*/; do cat $d/*.server.log | grep -c "patch=1356"; done | tr "\n" " ")"
        unset BIGCHERRY_PATCH_TRACE
        ;;
    3)
        echo "== step 3: Qwen3.8-27B no-regression"
        export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf
        export AR_FLAG="--allreduce cpu-root"
        rm -rf $R/nr27-*
        queue_one "VIS=0,1 SCRIPT nr27-1355 tools/lab/flash-next/prod27b-ab.sh @b-main2 @b-metamem-hcf1 $R/nr27-1355-out"
        queue_one "VIS=0,1 SCRIPT nr27-1358 tools/lab/flash-next/prod27b-ab.sh @b-main2 @b-metamem-sce1 $R/nr27-1358-out"
        queue_one "VIS=0,1 SCRIPT nr27-1356 tools/lab/flash-next/prod27b-ab.sh @b-main2 @b-metamem-mdw3 $R/nr27-1356-out BIGCHERRY_META_DISPATCH_THREADS=1 BIGCHERRY_PATCH_TRACE=1"
        for p in 1355 1358 1356; do
            echo "-- $p"
            grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]|md5" $R/nr27-$p.log | cut -c1-220
            md5sum $R/nr27-$p-out/*.txt 2> /dev/null | awk '{print $1}' | sort | uniq -c
        done
        unset AR_FLAG
        ;;
    4)
        echo "== step 4: QFP42 blocked-time trace, A = deferred catch-up on, B = off"
        export BC_MODEL=$FLASH BIGCHERRY_MTP_PROMPT_TIMING=1 BIGCHERRY_SUBMIT_TIMING=1
        rm -rf $R/q42s0-*
        AB_ENV="BIGCHERRY_MTP_DEFERRED_CATCHUP=0" bash tools/lab/flash-next/queue-env-ab.sh q42s0 b-metamem-mdw3 24576 98304 | grep -E "^==|^d[0-9]|^md5|SERVER_FAILED|blocked" | cut -c1-150
        for d in $R/q42s0-d*/*/; do
            echo "-- $d"
            # the prompt-fill line is the one with the largest nextn_block_ms
            grep -h BIGCHERRY_MTP_PROMPT_TIMING $d/*.server.log | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^nextn_block_ms=/) { split($i, a, "="); if (a[2] + 0 >= best) { best = a[2] + 0; line = $0 } }} END {print substr(line, 1, 700)}'
            grep -h BIGCHERRY_META_TIMING $d/*.server.log | awk '{for (i = 1; i <= NF; i++) { split($i, a, "="); v[a[1]] += a[2] } n++} END {printf "meta n=%d rebuild_ms=%.0f launch_ms=%.0f allreduce_ms=%.0f total_ms=%.0f\n", n, v["rebuild_us"]/1000, v["launch_us"]/1000, v["allreduce_us"]/1000, v["total_us"]/1000}'
        done
        unset BIGCHERRY_MTP_PROMPT_TIMING BIGCHERRY_SUBMIT_TIMING
        ;;
    *) echo "unknown step $s" ;;
    esac
done
echo FOLLOWUPS_DONE
