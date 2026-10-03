#!/bin/bash
# QFP13 / 1311 on profile v2: both arms 1307 (cache) + 1308 + 1309 + 1310 (row cap 64); new arm adds
# BIGCHERRY_HC_Q81=1 (hyper-connection pre-mix emits Q8_1). Quick ABA ~24K and ~80K (greedy identity at ~24K),
# census of the new arm, then a short trace of publishes vs misses.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1  # both arms; HC_Q81 only in the new arm
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1311 bigcherry:stock:linux-multi deploy-v2-plus-1311 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1311-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1311 @b-v2-1311 $R/flashnext-v2-1311-d24k BIGCHERRY_HC_Q81=1
VIS=0,1,2,3 SCRIPT v2-1311-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1311 @b-v2-1311 $R/flashnext-v2-1311-d80k BIGCHERRY_HC_Q81=1
VIS=0,1,2,3 SCRIPT v2-1311-census tools/lab/flash-next/census-run.sh @b-v2-1311 $R/flashnext-v2-1311-census BIGCHERRY_HC_Q81=1
VIS=0,1,2,3 SCRIPT v2-1311-trace tools/lab/flash-next/q81-trace-run.sh @b-v2-1311 $R/flashnext-v2-1311-trace BIGCHERRY_HC_Q81=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"  # flags only via the new-arm args: base arm must stay off
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1311-d24k v2-1311-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
echo "== greedy identity ~24K"
cmp -s $R/flashnext-v2-1311-d24k/base-a/timing.24576.greedy.txt $R/flashnext-v2-1311-d24k/new/timing.24576.greedy.txt && echo IDENTICAL || echo DIFFERENT
echo "== census"; grep -E "kernels/token|quantize|busy" $R/v2-1311-census.log | head -12
echo "== trace"; grep -E "^publish|misses in|miss ops" $R/v2-1311-trace.log
echo ALL_JOBS_DONE
