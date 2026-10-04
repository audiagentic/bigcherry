#!/bin/bash
# FMTP01 draft host overhead: 1317 + 1318 timing on all three arms of one diagnostic build (v4 + 1315-1318) at ~10K and ~80K;
# splits the serial fresh draft into draft-GPU work vs host (submit, sampling, hidden read). Waits for the calibration queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SPEC_TIMING=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-fmtp-calib.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-diag2 bigcherry:stock:linux-multi deploy-v4-diag gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT drafthost-d10k tools/lab/flash-next/quick-ab-depth.sh 10240 @b-diag2 @b-diag2 $R/flashnext-drafthost-d10k
VIS=0,1,2,3 SCRIPT drafthost-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-diag2 @b-diag2 $R/flashnext-drafthost-d80k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in 10k 80k; do
  echo "== gate0 d$d"; grep -E "^base-|^new|SERVER_FAILED" $R/drafthost-d$d.log
  python3 tools/lab/flash-next/spec-timing-summary.py $R/flashnext-drafthost-d$d/new/timing.server.log
  python3 tools/lab/flash-next/draft-timing-summary.py $R/flashnext-drafthost-d$d/new/timing.server.log
done
echo ALL_JOBS_DONE
