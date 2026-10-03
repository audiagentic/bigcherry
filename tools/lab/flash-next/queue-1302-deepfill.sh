#!/bin/bash
# 1302 check: production 192K config (-ts 2,2,3, q8_0 KV) filled to 131K depth (~164K tokens), where the deployment
# build OOMs in hipGraphInstantiate on the R9700 (flashnext-gather-ab-2/d131072). Pass = request completes and the
# server log shows BIGCHERRY_PATCH_HIT patch=1302_graph_oom_evict (or no OOM at all).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTKD=f16 CTVD=f16 DECODE_N=256 DEPTH=131072 CTX=196608 TS=2,2,3
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-deploy-1302b bigcherry:stock:linux-multi deploy-plus-1302 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-1302b-deepfill tools/lab/flash-next/long-ctx-profile.sh @b-deploy-1302b /mnt/data/bigcherry-work/runs/flashnext-1302-deepfill timing
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -h "BIGCHERRY_PATCH_HIT patch=1302\|hipGraphInstantiate\|^timing" /mnt/data/bigcherry-work/runs/flashnext-1302-deepfill/timing.server.log /mnt/data/bigcherry-work/runs/flashnext-1302-deepfill.log 2>/dev/null | head
echo ALL_JOBS_DONE
