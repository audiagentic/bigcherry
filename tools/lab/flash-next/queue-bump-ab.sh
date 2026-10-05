#!/bin/bash
# Pin-bump recheck of two existing build runs: Flash-Next 24K MTP decode ABA (base, new, base), then the
# 27B dual-XTX production ABBA at 10K and 32K (prod27b-ab.sh).
# Usage: queue-bump-ab.sh <base build run> <new build run> <run tag>
set -u
PREV=${1:?previous build run}
NEW=${2:?new build run}
TAG=${3:?run tag}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT $TAG-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @$PREV @$NEW $R/$TAG-d24k
VIS=0,1 SCRIPT $TAG-27b tools/lab/flash-next/prod27b-ab.sh @$PREV @$NEW $R/$TAG-27b
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== $NEW d24k (base = $PREV)"; grep -E "^base-|^new|SERVER_FAILED" $R/$TAG-d24k.log
md5sum $R/$TAG-d24k/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
echo "== 27B (A = $PREV, B = $NEW)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/$TAG-27b.log
echo ALL_JOBS_DONE
