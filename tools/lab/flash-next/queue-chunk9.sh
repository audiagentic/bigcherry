#!/bin/bash
# QFP22: 1007 (meta subgraph cgraph re-creation) + 1332: no-MTP crash check first, then 24K decode ABA and the 80K sweep.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-chunk8.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-chunk9 bigcherry:stock:linux-multi deploy-v6-plus-chunk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT chunk9-nomtp tools/lab/flash-next/chunk-nomtp.sh @b-chunk9 $R/qfp22-chunk9-nomtp
VIS=0,1,2,3 SCRIPT chunk9-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-chunk9 @b-chunk9 $R/qfp22-chunk9-d24k BIGCHERRY_QSA_CHUNK=512 UB=1024 B=1024
VIS=0,1,2,3 SCRIPT chunk9-ubc tools/lab/flash-next/ubchunk-sweep.sh @b-chunk9 $R/qfp22-chunk9-ubc 512:0 1024:512
JOBS
export BIGCHERRY_FEATURES=flashnext CTX=245760 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 EXTRA_OT='^token_embd\.weight$=CPU' B=512
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in d24k; do echo "== chunk9 $d"; grep -E "^base-|^new|SERVER_FAILED" $R/chunk9-$d.log; md5sum $R/qfp22-chunk9-$d/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c; done
echo "== sweep"; grep -E "^ub" $R/chunk9-ubc.log; grep -E "^chunk" $R/chunk9-nomtp.log; md5sum $R/qfp22-chunk9-nomtp/*/*.greedy.txt
echo ALL_JOBS_DONE
