#!/bin/bash
# QFP21 re-tests per GPT review: 1295 QSA gather on v6 at ~48K/~120K/~215K prompt tokens (240K f16); 1301 Q8_0 F32 widths on the 27B Q8_0 dual-XTX (MAXCOLS 1 vs 5).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext-v6
true
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-r1295 bigcherry:stock:linux-multi retest-1295 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-r1301 bigcherry:stock:linux-multi retest-1301 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT g-48k tools/lab/flash-next/gather-retest.sh 30720 @b-r1295 $R/qfp21-gather-48k
VIS=0,1,2,3 SCRIPT g-120k tools/lab/flash-next/gather-retest.sh 76800 @b-r1295 $R/qfp21-gather-120k
VIS=0,1,2,3 SCRIPT g-215k tools/lab/flash-next/gather-retest.sh 136000 @b-r1295 $R/qfp21-gather-215k
VIS=0,1 SCRIPT q8w-27b tools/lab/flash-next/prod27b-ab.sh @b-r1301 @b-r1301 $R/qfp21-1301-27b BIGCHERRY_Q8_F32_MAXCOLS=5 BIGCHERRY_PATCH_TRACE=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in g-48k g-120k g-215k; do
  echo "== 1295 $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log
  d=$R/qfp21-gather-${j#g-}; md5sum $d/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
  for a in base-a new base-b; do echo "$a vram: $(grep -h 'VRAM Total Used' $d/$a/timing.vram.txt 2>/dev/null | awk '{printf "%.1f ", $NF/1073741824}')GiB"; done
  grep -h "BIGCHERRY_PATCH_HIT patch=1295" $d/new/*.log | head -1
done
echo "== 1301 27B (A = MAXCOLS 1, B = MAXCOLS 5)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/q8w-27b.log
grep -hoE "BIGCHERRY_PATCH_HIT[^ ]*1301[^\n]*|BIGCHERRY_PATCH_HIT.*ncols[^ ]*" $R/qfp21-1301-27b/*.B.log | sort | uniq -c | head
echo ALL_JOBS_DONE
