#!/bin/bash
# QFP22: 1332 no-MTP crash diagnosis on b-chunk7 (after queue-chunk7).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-chunk7.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT chunk7-diag tools/lab/flash-next/chunk-nomtp-diag.sh /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/99ee3f8670d4d9d866377cd4ecafdd6e/d89e7735251562bbf62aabf884ab4103/bin/llama-server $R/qfp22-chunk7-diag
JOBS
export BIGCHERRY_FEATURES=flashnext CTX=245760 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 EXTRA_OT='^token_embd\.weight$=CPU' B=512
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
sed -n "1,200p" $R/chunk7-diag.log | grep -vE "PCIe"
echo ALL_JOBS_DONE
