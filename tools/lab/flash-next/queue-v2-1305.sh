#!/bin/bash
# QFP09 / 1305: expert share by bandwidth. Profile v2 placement (KV on both XTX) at -c 131072 so the XTX have VRAM
# headroom for extra expert share. Base = expert share follows -ts 0.31,0.27,0.42; new arms move experts off the
# R9700 via BIGCHERRY_FFN_TS. Quick ABA at ~24K and ~80K; the AR arrival skew is re-measured afterwards.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTX=131072 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1305 bigcherry:stock:linux-multi deploy-v2-plus-1305 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1305-34-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1305 @b-v2-1305 $R/flashnext-v2-1305-34-d24k BIGCHERRY_FFN_TS=0.34,0.33,0.33
VIS=0,1,2,3 SCRIPT v2-1305-37-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1305 @b-v2-1305 $R/flashnext-v2-1305-37-d24k BIGCHERRY_FFN_TS=0.37,0.35,0.28
VIS=0,1,2,3 SCRIPT v2-1305-34-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1305 @b-v2-1305 $R/flashnext-v2-1305-34-d80k BIGCHERRY_FFN_TS=0.34,0.33,0.33
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1305-34-d24k v2-1305-37-d24k v2-1305-34-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
echo ALL_JOBS_DONE
