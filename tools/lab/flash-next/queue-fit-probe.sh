#!/bin/bash
# v6 fit probe for -ub > 512 at f16 KV: R9700 OOMed at ub1024 with -ts 0.31,0.27,0.42 at 240K. Waits for the ub sweep.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-v6-ub.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-1327 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT fit-probe tools/lab/flash-next/fit-probe.sh @b-1327 $R/flashnext-fit-probe 768:0.31,0.27,0.42:245760 1024:0.33,0.28,0.39:245760 1024:0.33,0.28,0.39:229376 1024:0.34,0.29,0.37:212992 2048:0.34,0.29,0.37:196608
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^ub" $R/fit-probe.log
echo ALL_JOBS_DONE
