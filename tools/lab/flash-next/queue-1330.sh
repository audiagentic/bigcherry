#!/bin/bash
# QFP17 1330 (in-place QSA causal-mask add): (1) ub512 env screen off vs on at ~24K (greedy identity, speed);
# (2) ub sweep 512/1024 with 1330 on at 240K f16 (does ub1024 fit; prefill/decode), alloc-top logged.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1 BIGCHERRY_QSA_HOST_REMAP=1
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-1330b.log 2>/dev/null; do sleep 20; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-1330c bigcherry:stock:linux-multi deploy-v6-plus-1330 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT a1330c-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-1330c @b-1330c $R/flashnext-1330c-d24k BIGCHERRY_QSA_MASK_INPLACE=1
VIS=0,1,2,3 SCRIPT ub-1330c tools/lab/flash-next/ub-1330.sh @b-1330c $R/flashnext-ub-1330
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^base-|^new|SERVER_FAILED" $R/a1330c-d24k.log
D=$R/flashnext-1330c-d24k; md5sum $D/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c
grep -E "^ub" $R/ub-1330c.log
echo ALL_JOBS_DONE
