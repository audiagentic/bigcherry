#!/bin/bash
# QFP13 / 1308: rollback snapshot copies without CONT, on profile v2 with the 1307 Q8_1 cache on in BOTH arms.
# (1) greedy identity: NO_MTP is not usable here (rollback slots are MTP-only), so compare greedy MTP outputs of
#     both arms at ~24K (timing.*.greedy.txt); (2) quick ABA at ~24K and ~80K; (3) census of the new arm.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 GGML_HIP_Q8_1_CACHE_MODE=on
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1308 bigcherry:stock:linux-multi deploy-v2-plus-1308 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1308-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1308 @b-v2-1308 $R/flashnext-v2-1308-d24k BIGCHERRY_ROLLBACK_NO_CONT=1
VIS=0,1,2,3 SCRIPT v2-1308-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1308 @b-v2-1308 $R/flashnext-v2-1308-d80k BIGCHERRY_ROLLBACK_NO_CONT=1
VIS=0,1,2,3 SCRIPT v2-1308-census tools/lab/flash-next/census-run.sh @b-v2-1308 $R/flashnext-v2-1308-census BIGCHERRY_ROLLBACK_NO_CONT=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1308-d24k v2-1308-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
echo "== greedy identity ~24K (base-a vs new)"
cmp -s $R/flashnext-v2-1308-d24k/base-a/timing.24576.greedy.txt $R/flashnext-v2-1308-d24k/new/timing.24576.greedy.txt \
  && echo "IDENTICAL" || echo "DIFFERENT"
echo "== census"; grep -E "kernels/token|get/set_rows|busy" $R/v2-1308-census.log | head -12
echo ALL_JOBS_DONE
