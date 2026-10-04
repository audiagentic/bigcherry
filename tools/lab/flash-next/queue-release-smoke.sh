#!/bin/bash
# QFP18 follow-up: the promoted release build (bigcherry source + validated-enhancements now holding the Flash-Next stack) with BIGCHERRY_FEATURES=flashnext reproduces the pre-promotion v6 build (24K, ABA, greedy identity).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-maskref2.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-release bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT rel-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-release @b-v6 $R/flashnext-release-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^base-|^new|SERVER_FAILED" $R/rel-d24k.log
md5sum $R/flashnext-release-d24k/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
grep -h "BIGCHERRY_FEATURES flashnext" $R/flashnext-release-d24k/base-a/*.log | head -1
echo ALL_JOBS_DONE
