#!/bin/bash
# Flash-Next long-context prefill ABBA of two existing build runs at each given depth (production config:
# 240K ctx, f16 KV, flashnext profile, MTP draft on the 6900 XT). Places a small prefill difference without building.
# Usage: queue-prefill-ab.sh <build run A> <build run B> <tag> <depth>...
set -u
RUN_A=${1:?build run A}; RUN_B=${2:?build run B}; TAG=${3:?tag}; shift 3
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
for d in "$@"; do
    echo "VIS=0,1,2,3 SCRIPT $TAG-d$d tools/lab/flash-next/flash-prefill-ab.sh $d @$RUN_A @$RUN_B $R/$TAG-d$d" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== A = $RUN_A, B = $RUN_B"
for d in "$@"; do grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/$TAG-d$d.log; done
echo ALL_JOBS_DONE
