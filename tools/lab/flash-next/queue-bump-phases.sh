#!/bin/bash
# Pin 0504396 per-phase timing on v5 (+ 1317/1318/1319/1325) at ~10K: compare with the old-pin breakdown
# (draft 6.4 ms, target submit ~3.0 ms with 1326, sync ~27 ms; draft submit 0.53 ms/step) to locate the ~2% decode
# regression after the bump. Waits for the bump ABBA.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_SUBMIT_TIMING=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-bump-v5c.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v5diag-p2 bigcherry:stock:linux-multi deploy-v5-diag gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT phases-p2-d10k tools/lab/flash-next/quick-ab-depth.sh 10240 @b-v5diag-p2 @b-v5diag-p2 $R/flashnext-phases-p2-d10k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
F=$R/flashnext-phases-p2-d10k/new/timing.server.log
grep -E "^base-|^new|SERVER_FAILED" $R/phases-p2-d10k.log
python3 tools/lab/flash-next/spec-timing-summary.py $F | head -7
python3 tools/lab/flash-next/draft-timing-summary.py $F
python3 tools/lab/flash-next/submit-timing-summary.py $F
python3 tools/lab/flash-next/sched-split-summary.py $F
echo ALL_JOBS_DONE
