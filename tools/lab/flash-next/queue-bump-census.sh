#!/bin/bash
# Kernel census of v5 on the new pin (b-v5-p2c) at ~10K, to compare with the old pin (~1,000 kernels/token per XTX,
# v4 = v5 kernel-wise) and attribute the post-bump decode cost (#29819 dead-slot remap ops, #29824 mask construction,
# #29825 lightning indexer, #29184 shared-expert MMVQ fusion).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v5-p2c bigcherry:stock:linux-multi deploy-v5 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT census-p2 tools/lab/flash-next/census-run.sh @b-v5-p2c $R/flashnext-census-p2
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "kernels/token|busy|ms/tok" $R/census-p2.log | head -40
echo ALL_JOBS_DONE
