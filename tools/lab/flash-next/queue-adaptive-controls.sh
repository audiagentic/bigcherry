#!/bin/bash
# PRBE52 / 1268 controls on an EXISTING adaptive-mtp build run (nothing is built):
#   1. fixed depth 4 (A) vs fixed depth 3 (B) on the one binary, adaptive off - does the greedy text depend on the
#      draft depth on this stack at all? (if yes, the adaptive arm's text difference is not a 1268 defect)
#   2. adaptive off vs the production build at the given depths (clean replacement of a withdrawn lane).
# Usage: queue-adaptive-controls.sh <tag> <adaptive build run> <production build run> <depth>...
set -u
TAG=${1:?tag}; RUN=${2:?adaptive build run}; PROD=${3:?production build run}; shift 3
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_RUNS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext SPEC_N=${SPEC_MAX:-4} BIGCHERRY_PATCH_TRACE=1
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
: > "$jobs"
for d in "${depths[@]}"; do
  echo "VIS=0,1,2,3 SCRIPT adaptctl-$TAG-depth-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/adaptctl-$TAG-depth-d$d SPEC_N=3" >> "$jobs"
  echo "VIS=0,1,2,3 SCRIPT adaptctl-$TAG-off-d$d tools/lab/flash-next/flash-prefill-ab.sh $d @$PROD @$RUN $R/adaptctl-$TAG-off-d$d" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "${depths[@]}"; do
  echo "== depth $d: A = fixed depth 4, B = fixed depth 3 (same binary, adaptive off)"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/adaptctl-$TAG-depth-d$d.log
  echo "== depth $d: adaptive off, A = $PROD (production), B = $RUN"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/adaptctl-$TAG-off-d$d.log
done
echo ALL_JOBS_DONE
