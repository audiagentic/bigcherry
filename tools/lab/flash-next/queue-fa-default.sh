#!/bin/bash
# 1334 default-on check: build the production set (sparse flash attention on unless BIGCHERRY_FA_SPARSE=0), then
#   1. Qwen3.8-27B dual-XTX production ABBA (prod27b-ab.sh) against a build where the path is off: a model with no
#      n_kv_max must give the same text and speed;
#   2. Flash-Next prefill ABBA on the new binary, default vs BIGCHERRY_FA_SPARSE=0: the off switch still works.
# Usage: queue-fa-default.sh <tag> <reference build run with the path off> <flash depth> [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; REF=${2:?reference build run}; DEPTH=${3:?flash depth}; shift 3
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-fadef-$TAG
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT fadef-$TAG-27b tools/lab/flash-next/prod27b-ab.sh @$REF @$RUN $R/fadef-$TAG-27b
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_FEATURES=flashnext
echo "VIS=0,1,2,3 SCRIPT fadef-$TAG-flash tools/lab/flash-next/flash-prefill-env-ab.sh $DEPTH @$RUN $R/fadef-$TAG-flash BIGCHERRY_FA_SPARSE=0" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== 27B: A = $REF (path off), B = $RUN (default on)"; grep -E "^d[0-9]|greedy|^ +[0-9]+ |SERVER_FAILED" $R/fadef-$TAG-27b.log
echo "== Flash-Next: A = default, B = BIGCHERRY_FA_SPARSE=0"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/fadef-$TAG-flash.log
echo ALL_JOBS_DONE
