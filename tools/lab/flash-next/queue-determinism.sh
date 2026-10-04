#!/bin/bash
# QFP15: same build, same ~24K prompt, three runs (quick-ab-depth base-a/new/base-b all on one binary and one env) with
# BIGCHERRY_DRAFT_TRACE=1 (1315). Diff the BIGCHERRY_DRAFT_TRACE streams: first divergence = accept target_h (target
# nondeterministic) or step p/h (draft nondeterministic). Waits for the 1312c queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_DRAFT_TRACE=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-v3-1312c.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-det bigcherry:stock:linux-multi deploy-v3-determinism gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT det-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-det @b-det $R/flashnext-det-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
D=$R/flashnext-det-d24k
grep -E "^base-|^new|SERVER_FAILED" $R/det-d24k.log
for a in base-a new base-b; do grep -o "BIGCHERRY_DRAFT_TRACE.*" $D/$a/timing.server.log > $D/$a.trace 2>/dev/null; echo "$a: $(wc -l < $D/$a.trace) trace lines"; done
for pair in "base-a new" "base-a base-b"; do set -- $pair
  echo "== first divergence $1 vs $2"
  diff <(cat -n $D/$1.trace) <(cat -n $D/$2.trace) | head -6
done
echo ALL_JOBS_DONE
