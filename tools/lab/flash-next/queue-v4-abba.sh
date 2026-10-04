#!/bin/bash
# QFN01 v4 adoption: profile v3 (b-v3c) vs v4 = v3 + 1312 + 1313 (b-v4, BIGCHERRY_SCALE_ACT_FUSE=1 on the new arm),
# ABBA x2 at ~8K and ~64K, 512 decode tokens per arm. Waits for the Gate 0 queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-gate0.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v3c bigcherry:stock:linux-multi deploy-v2-plus-1311 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-v4 bigcherry:stock:linux-multi deploy-v4 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v4-abba tools/lab/flash-next/abba-depths.sh @b-v3c @b-v4 $R/flashnext-v4-abba BIGCHERRY_SCALE_ACT_FUSE=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^d[0-9]+ " $R/v4-abba.log
echo ALL_JOBS_DONE
