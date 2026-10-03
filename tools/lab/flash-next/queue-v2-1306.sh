#!/bin/bash
# QFP12 / 1306: Qwen4Exp shared-expert split on production profile v2 (f16/f16 240K, KV on both XTX).
# Same binary both arms: base = mirrored shared expert (upstream), new = BIGCHERRY_SHEXP_SPLIT=1. Quick ABA at ~24K
# and ~80K, then a VRAM snapshot of each arrangement.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1306 bigcherry:stock:linux-multi deploy-v2-plus-1306 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1306-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1306 @b-v2-1306 $R/flashnext-v2-1306-d24k BIGCHERRY_SHEXP_SPLIT=1
VIS=0,1,2,3 SCRIPT v2-1306-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1306 @b-v2-1306 $R/flashnext-v2-1306-d80k BIGCHERRY_SHEXP_SPLIT=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1306-d24k v2-1306-d80k; do echo "== $j"; grep -E "^base-|^new|SERVER_FAILED" $R/$j.log; done
for arm in base-a new; do
  echo "== vram d24k $arm"; tail -n 4 $R/flashnext-v2-1306-d24k/$arm/timing.vram.txt 2>/dev/null
  grep -h "BIGCHERRY_PATCH_HIT patch=1306" $R/flashnext-v2-1306-d24k/$arm/timing.server.log 2>/dev/null | head -1
done
echo ALL_JOBS_DONE
