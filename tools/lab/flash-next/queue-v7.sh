#!/bin/bash
# Candidate v7 = v6 + 1295 gather + 1332 chunks (ub1024, chunk 512): decode ABA at ~38K and ~120K tokens, prefill + decode sweep at the 80K fill (after chunk-confirm).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-chunk-confirm.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v7 bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v7-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v7 @b-v7 $R/qfp22-v7-d24k BIGCHERRY_QSA_GATHER=1 BIGCHERRY_QSA_CHUNK=512 UB=1024 B=1024
VIS=0,1,2,3 SCRIPT v7-d120k tools/lab/flash-next/quick-ab-depth.sh 76800 @b-v7 @b-v7 $R/qfp22-v7-d120k BIGCHERRY_QSA_GATHER=1 BIGCHERRY_QSA_CHUNK=512 UB=1024 B=1024
VIS=0,1,2,3 SCRIPT v7-sweep tools/lab/flash-next/v7-sweep.sh @b-v7 $R/qfp22-v7-sweep
JOBS
export BIGCHERRY_FEATURES=flashnext CTX=245760 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 EXTRA_OT='^token_embd\.weight$=CPU' B=512
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in d24k d120k; do echo "== v7 $d"; grep -E "^base-|^new|SERVER_FAILED" $R/v7-$d.log; md5sum $R/qfp22-v7-$d/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c; done
echo "== v7 sweep"; grep -E "^ub" $R/v7-sweep.log
echo ALL_JOBS_DONE
