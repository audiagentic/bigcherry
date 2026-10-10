#!/bin/bash
# Prefill attribution for the next patch round (QFP17): rocprofv3 kernel trace + stats of one uncached Flash-Next
# prefill at each depth on an existing build run (long-ctx-profile.sh prefillprof), production config, then a
# per-kernel time table per device so the levers can be ranked (QSA attention / lightning indexer / MoE MMQ / GDN /
# AllReduce and copies).
# Usage: queue-prefill-profile.sh <build run> <tag> <depth>... [wait=<log with ALL_JOBS_DONE>]
set -u
RUN=${1:?build run}; TAG=${2:?tag}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
DEPTHS=()
for a in "$@"; do
    case "$a" in
        wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;;
        *) DEPTHS+=("$a") ;;
    esac
done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
for d in "${DEPTHS[@]}"; do
    jobs=$(mktemp)
    echo "VIS=0,1,2,3 SCRIPT $TAG-d$d tools/lab/flash-next/long-ctx-profile.sh @$RUN $R/$TAG-d$d ${MODE:-prefillprof}" > "$jobs"
    DEPTH=$d bash tools/lab/plan-qualification/queue.sh "$jobs"
    echo "QUEUE_EXIT=$? depth=$d $(date -Is)"
    rm -f "$jobs"
    echo "== $TAG depth $d"
    grep -hE "^timing:|prefill|SERVER_FAILED" $R/$TAG-d$d.log | tail -4
    if [ "${MODE:-prefillprof}" = prefillprof ]; then
        python3 tools/lab/flash-next/prefill-kernel-table.py $R/$TAG-d$d/rocprof
    else
        if [ -f $R/$TAG-d$d/sync-sites.txt ]; then cut -c1-230 $R/$TAG-d$d/sync-sites.txt | head -150; else grep -E "^ +[0-9]" $R/$TAG-d$d.log | cut -c1-200 | head -72; fi
    fi
done
echo ALL_JOBS_DONE
