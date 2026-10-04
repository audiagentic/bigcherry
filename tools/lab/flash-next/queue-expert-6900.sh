#!/bin/bash
# Expert-to-6900 sweep on v6 (pin 0504396): decode cost of moving 2/4/6/8 layers' routed experts to the 6900 at ub512,
# and the prefill gain of the larger ubatch that frees: ub1024 (4/6 layers), ub2048 (6/8 layers). Baselines first/last.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 DECODE_N=512
export BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
L2='20|44'; L4='8|20|32|44'; L6='4|12|20|28|36|44'; L8='4|10|16|22|28|34|40|46'
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-1327 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT expert-6900 tools/lab/flash-next/expert-6900-sweep.sh @b-1327 $R/flashnext-expert-6900 none:512 $L2:512 $L4:512 $L6:512 $L4:1024 $L6:1024 $L6:2048 $L8:2048 none:512
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^run" $R/expert-6900.log
echo ALL_JOBS_DONE
