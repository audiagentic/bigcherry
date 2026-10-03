#!/bin/bash
# Diagnose why 1309's published RMSNorm Q8_1 activations are not hit by MMVQ: one short ~8K request on profile v2
# with 1307/1308/1309/1310 on and BIGCHERRY_Q81_TRACE=1, then pair publish-* lines with 1307 miss lines
# (same generation? same node? same data/shape?).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=65536 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-q81trace bigcherry:stock:linux-multi deploy-v2-plus-1310 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-q81trace tools/lab/flash-next/q81-trace-run.sh @b-v2-q81trace $R/flashnext-v2-q81trace
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
rm -f "$jobs"
tail -n 40 $R/v2-q81trace.log
echo ALL_JOBS_DONE
