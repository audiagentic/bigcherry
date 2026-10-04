#!/bin/bash
# QFP17 1332 third pass (chunk only when n_tokens > chunk): MTP identity at 24K, no-MTP single-token decode, ub1024 fit.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext-v6
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-release-smoke.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-chunk3 bigcherry:stock:linux-multi deploy-v6-plus-chunk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT a-chunk3-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-chunk3 @b-chunk3 $R/flashnext-chunk3-d24k BIGCHERRY_QSA_CHUNK=256
VIS=0,1,2,3 SCRIPT chunk3-nomtp tools/lab/flash-next/chunk-nomtp.sh @b-chunk3 $R/flashnext-chunk3-nomtp
VIS=0,1,2,3 SCRIPT ubc3 tools/lab/flash-next/ubchunk-sweep.sh @b-chunk3 $R/flashnext-ubchunk3 512:0 1024:256
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^base-|^new|SERVER_FAILED" $R/a-chunk3-d24k.log
md5sum $R/flashnext-chunk3-d24k/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
grep -E "^chunk|^ +[0-9]" $R/chunk3-nomtp.log
grep -E "^ub" $R/ubc3.log
echo ALL_JOBS_DONE
