#!/bin/bash
# FMTP03: MTP look-ahead with promotion gated on the tail's lowest draft probability (BIGCHERRY_MTP_AHEAD_PROMOTE_P).
# Builds deploy-v6-plus-ahead as b-ahead-<tag>, then for each threshold an ahead-ab.sh ABA (base / look-ahead /
# base, same binary) at 24K and 80K. Reports ms/step, t/s, acceptance, per-position acceptance, greedy md5 counts
# and the look-ahead stats (promoted rounds) per run.
# Usage: queue-ahead-gate.sh <tag> <threshold>... [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}
shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
THR=()
for a in "$@"; do
    case "$a" in
        wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;;
        *) THR+=("$a") ;;
    esac
done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD b-ahead-$TAG bigcherry:stock:linux-multi deploy-v6-plus-ahead gfx1100,gfx1201,gfx1030" > "$jobs"
for t in "${THR[@]}"; do
    for d in 24576 81920; do
        echo "VIS=0,1,2,3 SCRIPT gate-$TAG-p$t-d$d tools/lab/flash-next/ahead-ab.sh $d @b-ahead-$TAG $R/gate-$TAG-p$t-d$d BIGCHERRY_MTP_AHEAD_PROMOTE_P=$t" >> "$jobs"
    done
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for t in "${THR[@]}"; do
    for d in 24576 81920; do
        j=gate-$TAG-p$t-d$d
        echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log
        md5sum $R/$j/*/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c
        for a in base-a new base-b; do echo "  $a $(grep -h 'acc per pos' $R/$j/$a/*.log 2>/dev/null | tail -1 | sed 's/.*acc per pos/acc per pos/')"; done
        grep -h BIGCHERRY_MTP_AHEAD $R/$j/new/*.log 2>/dev/null | tail -1 | sed 's/.*BIGCHERRY_MTP_AHEAD/  AHEAD/'
    done
done
echo ALL_JOBS_DONE
