#!/bin/bash
# FMTP03: MTP-ahead tail confidence cut (BIGCHERRY_MTP_AHEAD_PMIN 0.85 / 0.95) at 24K and 80K, 240K f16; same binary ABA.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext-v6
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-chunk.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-ahead2 bigcherry:stock:linux-multi deploy-v6-plus-ahead gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT ap85-d24k tools/lab/flash-next/ahead-ab.sh 24576 @b-ahead2 $R/flashnext-ap85-d24k 0.85
VIS=0,1,2,3 SCRIPT ap95-d24k tools/lab/flash-next/ahead-ab.sh 24576 @b-ahead2 $R/flashnext-ap95-d24k 0.95
VIS=0,1,2,3 SCRIPT ap85-d80k tools/lab/flash-next/ahead-ab.sh 81920 @b-ahead2 $R/flashnext-ap85-d80k 0.85
VIS=0,1,2,3 SCRIPT ap95-d80k tools/lab/flash-next/ahead-ab.sh 81920 @b-ahead2 $R/flashnext-ap95-d80k 0.95
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in ap85-d24k ap95-d24k ap85-d80k ap95-d80k; do
  echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log
  md5sum $R/flashnext-$j/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
  grep -h BIGCHERRY_MTP_AHEAD $R/flashnext-$j/new/*.log | tail -1
done
echo ALL_JOBS_DONE
