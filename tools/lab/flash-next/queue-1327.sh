#!/bin/bash
# 1327 env screen on one v5+1327 build (new pin): base off, new BIGCHERRY_QSA_HOST_REMAP=1; ~24K and ~80K, greedy identity, census.
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
VIS=0,1,2,3 BUILD b-1327 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT a1327-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-1327 @b-1327 $R/flashnext-1327-d24k BIGCHERRY_QSA_HOST_REMAP=1
VIS=0,1,2,3 SCRIPT a1327-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-1327 @b-1327 $R/flashnext-1327-d80k BIGCHERRY_QSA_HOST_REMAP=1
VIS=0,1,2,3 SCRIPT census-1327 tools/lab/flash-next/census-run.sh @b-1327 $R/flashnext-census-1327 BIGCHERRY_QSA_HOST_REMAP=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in 24k 80k; do echo "== d$d"; grep -E "^base-|^new|SERVER_FAILED" $R/a1327-d$d.log; D=$R/flashnext-1327-d$d; t0=$(ls $D/base-a/*.greedy.txt | head -1); for f in $D/*/*.greedy.txt; do cmp -s "$t0" "$f" && echo "$(basename $(dirname $f)) IDENTICAL" || echo "$(basename $(dirname $f)) DIFFERENT"; done; done
grep -E "kernels/token|elementwise|get/set" $R/census-1327.log | head -6
echo ALL_JOBS_DONE
