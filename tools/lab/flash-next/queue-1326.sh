#!/bin/bash
# QFP16 1326: async host->device split inputs. One build (v4 + 1326 + timing diagnostics), env screen: base arms off,
# new arm BIGCHERRY_SCHED_ASYNC_INPUTS=1; SPEC/SUBMIT timing on all arms (overhead nil). ~24K and ~80K, greedy identity,
# then the 1325/1320/1317 breakdown of the new arm. Waits for the sched-split queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_SUBMIT_TIMING=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-sched-split.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-1326 bigcherry:stock:linux-multi deploy-v4-plus-1326-diag gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT a1326-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-1326 @b-1326 $R/flashnext-1326-d24k BIGCHERRY_SCHED_ASYNC_INPUTS=1
VIS=0,1,2,3 SCRIPT a1326-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-1326 @b-1326 $R/flashnext-1326-d80k BIGCHERRY_SCHED_ASYNC_INPUTS=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in 24k 80k; do
  D=$R/flashnext-1326-d$d
  echo "== d$d"; grep -E "^base-|^new|SERVER_FAILED" $R/a1326-d$d.log
  t0=$(ls $D/base-a/*.greedy.txt | head -1); for f in $D/*/*.greedy.txt; do cmp -s "$t0" "$f" && echo "$(basename $(dirname $f)) IDENTICAL" || echo "$(basename $(dirname $f)) DIFFERENT"; done
  for a in base-a new; do echo "-- $a"; python3 tools/lab/flash-next/spec-timing-summary.py $D/$a/timing.server.log | sed -n 2,4p
    python3 tools/lab/flash-next/sched-split-summary.py $D/$a/timing.server.log; done
done
echo ALL_JOBS_DONE
