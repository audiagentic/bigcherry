#!/bin/bash
# QFP25 / 1334: build the production set + 1334, then on that one binary compare BIGCHERRY_FA_SPARSE off / on:
# prefill ABBA at each depth (flash-prefill-env-ab.sh), a 24K MTP decode ABA (quick-ab-depth.sh) and the no-MTP
# output-distribution gate (flash-probs-ab.sh).
# Usage: queue-sparse-fa.sh <tag> <depth>... [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; shift
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
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-sparsefa-$TAG
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi sparse-fa gfx1100,gfx1201,gfx1030" > "$jobs"
for d in "${DEPTHS[@]}"; do
    echo "VIS=0,1,2,3 SCRIPT sparsefa-$TAG-pp$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/sparsefa-$TAG-pp$d BIGCHERRY_FA_SPARSE=1" >> "$jobs"
done
echo "VIS=0,1,2,3 SCRIPT sparsefa-$TAG-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @$RUN @$RUN $R/sparsefa-$TAG-d24k BIGCHERRY_FA_SPARSE=1" >> "$jobs"
# output-distribution gate without MTP: 24K depth (38.7K tokens; the sparse path starts at ~32.8K cells, and a CPU f32
# reference text exists for this prompt) and the first prefill depth
for d in 24576 "${DEPTHS[0]}"; do
    echo "VIS=0,1,2,3 SCRIPT sparsefa-$TAG-probs$d tools/lab/flash-next/flash-probs-ab.sh $d @$RUN $R/sparsefa-$TAG-probs$d BIGCHERRY_FA_SPARSE=1" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "error:|Error|FAILED" $R/$RUN.log | head -12
for d in "${DEPTHS[@]}"; do echo "== prefill depth $d (A = sparse off, B = on)"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/sparsefa-$TAG-pp$d.log; done
echo "== 24K MTP decode (base = off, new = on)"; grep -E "^base-|^new|SERVER_FAILED" $R/sparsefa-$TAG-d24k.log
md5sum $R/sparsefa-$TAG-d24k/*/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c
for d in 24576 "${DEPTHS[0]}"; do echo "== output distributions, no MTP, depth $d (A = sparse off, B = on)"; grep -E "^A1|^B:|^A2|floor|change|text md5|PROBS_MISSING" $R/sparsefa-$TAG-probs$d.log; done
echo ALL_JOBS_DONE
