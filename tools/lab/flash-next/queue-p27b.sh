#!/bin/bash
# QFP18: 27B dual-XTX no-regression re-run with each build's default all-reduce (base build lacks 1291 cpu-root).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext-v6
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-ahead-pmin.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-base bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-v6 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT p27b-2 tools/lab/flash-next/prod27b-ab.sh @b-base @b-v6 $R/prod27b-promote-2
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/p27b-2.log
echo ALL_JOBS_DONE
