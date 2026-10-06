#!/bin/bash
# PRBE52 follow-up: upstream's own confidence gate for MTP drafting. draft-mtp stops a draft when the head's top-1
# probability falls below --spec-draft-p-min (LLAMA_ARG_SPEC_DRAFT_P_MIN, default 0 = never). With a higher depth cap
# this is an adaptive depth per round with no controller: long drafts where the head is sure, short ones where it is
# not. Flash-Next production on an EXISTING build run; per setting and depth a decode ABBA, A = fixed depth 3
# (production), B = the setting.
# Usage: queue-pmin-screen.sh <tag> <build run> <depth>... [wait=<chain log with ALL_RUNS_DONE>]
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
SETTINGS=("LLAMA_ARG_SPEC_DRAFT_P_MIN=0.6 SPEC_N=5" "LLAMA_ARG_SPEC_DRAFT_P_MIN=0.75 SPEC_N=5" "LLAMA_ARG_SPEC_DRAFT_P_MIN=0.9 SPEC_N=6" "LLAMA_ARG_SPEC_DRAFT_P_MIN=0.75 SPEC_N=3")
# SETTINGS_LIST overrides the list: settings separated by ";" (e.g. adaptive depth on a build that carries 1268)
if [ -n "${SETTINGS_LIST:-}" ]; then IFS=";" read -r -a SETTINGS <<< "$SETTINGS_LIST"; fi
jobs=$(mktemp)
: > "$jobs"
for d in "${depths[@]}"; do
  i=0
  for setting in "${SETTINGS[@]}"; do
    i=$((i+1))
    echo "VIS=0,1,2,3 SCRIPT pmin-$TAG-d$d-$i tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/pmin-$TAG-d$d-$i $setting" >> "$jobs"
  done
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "${depths[@]}"; do
  i=0
  for setting in "${SETTINGS[@]}"; do
    i=$((i+1))
    echo "== depth $d: A = fixed depth 3, B = $setting"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/pmin-$TAG-d$d-$i.log
  done
done
echo ALL_JOBS_DONE
