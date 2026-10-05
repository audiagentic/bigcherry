#!/bin/bash
# Pin-bump smoke for the Flash-Next deployment: build the deploy recipe on the new pin, then
# no-MTP run (1007 + 1332 chunk 256), and a 24K MTP decode ABA of the previous pin's build vs the new one.
# Usage: queue-bump-smoke.sh <previous build run, e.g. b-chunk8> <new build run, e.g. b-bump-b11402>
set -u
PREV=${1:?previous build run}
NEW=${2:?new build run}
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
VIS=0,1,2,3 BUILD $NEW bigcherry:stock:linux-multi deploy-v6-plus-chunk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT $NEW-nomtp tools/lab/flash-next/chunk-nomtp.sh @$NEW $R/$NEW-nomtp
VIS=0,1,2,3 SCRIPT $NEW-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @$PREV @$NEW $R/$NEW-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== $NEW d24k (base = $PREV)"; grep -E "^base-|^new|SERVER_FAILED" $R/$NEW-d24k.log
md5sum $R/$NEW-d24k/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
echo "== no-MTP"; grep -E "^chunk" $R/$NEW-nomtp.log; md5sum $R/$NEW-nomtp/*/*.greedy.txt
echo ALL_JOBS_DONE
