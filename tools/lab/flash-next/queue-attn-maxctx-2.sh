#!/bin/bash
# 1303 round 2 (KV pinned to XTX0 + R9700, BIGCHERRY_ATTN_TS=1,0,1 unrotated): f16/q8_0 past 208K to 256K, and f16/f16.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-deploy-1303 bigcherry:stock:linux-multi deploy-plus-1302-1303 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT attn-maxctx-101-hi tools/lab/flash-next/attn-maxctx.sh @b-deploy-1303 $R/flashnext-attn-maxctx-101-hi 1,0,1 0 212992 262144 f16 q8_0
VIS=0,1,2,3 SCRIPT attn-maxctx-101-f16 tools/lab/flash-next/attn-maxctx.sh @b-deploy-1303 $R/flashnext-attn-maxctx-101-f16 1,0,1 0 131072 262144 f16 f16
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for t in 101-hi 101-f16; do echo "== $t"; grep -h "RESULT\|fails" $R/attn-maxctx-$t.log 2>/dev/null; done
echo ALL_JOBS_DONE
