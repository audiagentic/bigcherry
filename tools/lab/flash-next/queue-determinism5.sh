#!/bin/bash
# QFP15 run 5: is upstream itself run-to-run nondeterministic on this lane? Stock build (no BigCherry patches), RCCL
# AllReduce (AR=auto), no attention split, CTX 65536 (f16 KV fits without 1303), three runs of one binary at ~24K;
# compare drafted/accepted counts, the warm-up acceptance lines and the greedy text. Waits for the meta-timing queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=${BC_TMP_ROOT}/bc-rocm
export BC_MODEL=${BC_MODEL_ROOT}/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=${BC_MODEL_ROOT}/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=65536 TS=0.31,0.27,0.42 B=512 AR=auto
export EXTRA_OT='^token_embd\.weight$=CPU'
until grep -q ALL_JOBS_DONE ${BC_WORK_ROOT}/runs/queue-meta-timing.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=${BC_WORK_ROOT}/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-stock-none bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT det5-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-stock-none @b-stock-none $R/flashnext-det5-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
D=$R/flashnext-det5-d24k
grep -E "^base-|^new|SERVER_FAILED" $R/det5-d24k.log
for a in base-a new base-b; do
  echo "== $a: $(grep -m1 'draft acceptance' $D/$a/timing.server.log | sed -E 's/.*draft acceptance/draft acceptance/') | text $(md5sum < $D/$a/timing.24576.greedy.txt | cut -c1-8)"
done
echo ALL_JOBS_DONE
