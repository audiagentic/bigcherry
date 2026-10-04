#!/bin/bash
# QFP13 1312: same-shape MUL emits Q8_1 (attn gate + GDN gated norm). Build A/B: v3 (1311 build) vs v3+1312, same flags on both.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v3 bigcherry:stock:linux-multi deploy-v2-plus-1311 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-v3-1312 bigcherry:stock:linux-multi deploy-v3-plus-1312 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v3-1312-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v3 @b-v3-1312 $R/flashnext-v3-1312-d24k
VIS=0,1,2,3 SCRIPT v3-1312-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v3 @b-v3-1312 $R/flashnext-v3-1312-d80k
VIS=0,1,2,3 SCRIPT v3-1312-census tools/lab/flash-next/census-run.sh @b-v3-1312 $R/flashnext-v3-1312-census
VIS=0,1,2,3 SCRIPT v3-1312-trace tools/lab/flash-next/q81-trace-run.sh @b-v3-1312 $R/flashnext-v3-1312-trace
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v3-1312-d24k v3-1312-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
echo "== greedy identity ~24K"
cmp -s $R/flashnext-v3-1312-d24k/base-a/timing.24576.greedy.txt $R/flashnext-v3-1312-d24k/new/timing.24576.greedy.txt && echo IDENTICAL || echo DIFFERENT
echo "== census"; grep -E "kernels/token|quantize|busy" $R/v3-1312-census.log | head -12
echo "== trace"; grep -E "^publish|misses in|miss ops|publish-mul" $R/v3-1312-trace.log
echo ALL_JOBS_DONE
