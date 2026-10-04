#!/bin/bash
# QFP15 run 4: v3/v4 runtime flags OFF (Q8_1 cache, RMS/ACT/HC producers, rollback no-CONT, scale-act fusion), profile v2
# flags only, three runs of one build with 1315 trace. Deterministic => one of the flag-gated patches is the cause. Waits for calib2.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=off
export BIGCHERRY_DRAFT_TRACE=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-fmtp-calib2.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-det bigcherry:stock:linux-multi deploy-v3-determinism gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT det4-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-det @b-det $R/flashnext-det4-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
D=$R/flashnext-det4-d24k
grep -E "^base-|^new|SERVER_FAILED" $R/det4-d24k.log
for a in base-a new base-b; do grep -o "BIGCHERRY_DRAFT_TRACE.*" $D/$a/timing.server.log > $D/$a.trace; echo "$a: $(wc -l < $D/$a.trace) lines, first $(head -1 $D/$a.trace | cut -c1-60)"; done
for pair in "base-a new" "base-a base-b" "new base-b"; do set -- $pair
  echo "== $1 vs $2: $(cmp $D/$1.trace $D/$2.trace >/dev/null && echo IDENTICAL || echo "diverge at line $(paste -d'|' $D/$1.trace $D/$2.trace | awk -F'|' '$1 != $2 {print NR; exit}')")"
done
echo ALL_JOBS_DONE
