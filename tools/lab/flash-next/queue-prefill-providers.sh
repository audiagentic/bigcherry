#!/bin/bash
# Queue wrapper for prefill-provider-sweep.sh on an existing build run (production config, 240K f16).
# Usage: queue-prefill-providers.sh <build run> <tag> <depth> <provider>... [wait=<log with ALL_JOBS_DONE>]
set -u
RUN=${1:?build run}; TAG=${2:?tag}; DEPTHV=${3:?depth}; shift 3
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
PROV=()
for a in "$@"; do
    case "$a" in
        wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;;
        *) PROV+=("$a") ;;
    esac
done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=64
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT $TAG tools/lab/flash-next/prefill-provider-sweep.sh $DEPTHV @$RUN $R/$TAG ${PROV[*]}" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^ar=|^md5|SERVER_FAILED" $R/$TAG.log
echo ALL_JOBS_DONE
