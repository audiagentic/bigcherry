#!/bin/bash
# QFP13 / 1307: Q8_1 activation reuse on production profile v2 (f16/f16 240K, KV on both XTX).
# Same binary both arms: base = cache off, new = GGML_HIP_Q8_1_CACHE_MODE=on. Quick ABA at ~24K and ~80K, then a
# short decode-mode profile of the cache-on arm at ~10K for the kernel census (quantize_q8_1 per token = activation).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1307b bigcherry:stock:linux-multi deploy-v2-plus-1307 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1307b-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1307b @b-v2-1307b $R/flashnext-v2-1307-d24k GGML_HIP_Q8_1_CACHE_MODE=on
VIS=0,1,2,3 SCRIPT v2-1307b-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1307b @b-v2-1307b $R/flashnext-v2-1307-d80k GGML_HIP_Q8_1_CACHE_MODE=on
VIS=0,1,2,3 SCRIPT v2-1307b-census tools/lab/flash-next/census-run.sh @b-v2-1307b $R/flashnext-v2-1307-census GGML_HIP_Q8_1_CACHE_MODE=on
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1307b-d24k v2-1307b-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
echo "== census"; grep -E "kernels/token|quantize|busy" $R/v2-1307b-census.log | head -12
echo ALL_JOBS_DONE
