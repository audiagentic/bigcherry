#!/bin/bash
# PGC14: RCCL algorithm / protocol screen for the tensor-split all-reduce (ncclDevKernel_Generic_4, the largest prefill
# kernel family). Flash-Next production on an EXISTING build run; per setting a prefill ABBA at the given depth,
# A = default RCCL, B = the setting. NCCL_DEBUG=INFO in arm B so the server log shows what RCCL actually selected.
# Usage: queue-rccl-screen.sh <tag> <build run> <depth> [wait=<chain log with ALL_RUNS_DONE>]
set -u
TAG=${1:?tag}; RUN=${2:?build run}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_RUNS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
d=${depths[0]}
jobs=$(mktemp)
: > "$jobs"
i=0
for setting in "NCCL_PROTO=Simple" "NCCL_PROTO=LL" "NCCL_PROTO=LL128" "NCCL_ALGO=Tree" "NCCL_ALGO=Ring"; do
  i=$((i+1))
  echo "VIS=0,1,2,3 SCRIPT rccl-$TAG-$i tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/rccl-$TAG-$i $setting NCCL_DEBUG=INFO" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
i=0
for setting in "NCCL_PROTO=Simple" "NCCL_PROTO=LL" "NCCL_PROTO=LL128" "NCCL_ALGO=Tree" "NCCL_ALGO=Ring"; do
  i=$((i+1))
  echo "== $setting (B) vs default (A)"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/rccl-$TAG-$i.log
  echo "   RCCL in B: $(grep -rhoE "NCCL INFO[^
]*(Algo|Proto|algorithm|protocol)[^
]*" $R/rccl-$TAG-$i/*-B/*.server.log 2>/dev/null | cut -c1-140 | sort | uniq -c | sort -rn | head -3 | tr '
' ';')"
done
echo ALL_JOBS_DONE
