#!/bin/bash
# 1303: f16-K/q8_0-V max context with KV heads pinned (unrotated) to each device pair; -ts adapts to fill VRAM.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-deploy-1303 bigcherry:stock:linux-multi deploy-plus-1302-1303 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT attn-maxctx-101 tools/lab/flash-next/attn-maxctx.sh @b-deploy-1303 $R/flashnext-attn-maxctx-101 1,0,1
VIS=0,1,2,3 SCRIPT attn-maxctx-110 tools/lab/flash-next/attn-maxctx.sh @b-deploy-1303 $R/flashnext-attn-maxctx-110 1,1,0
VIS=0,1,2,3 SCRIPT attn-maxctx-011 tools/lab/flash-next/attn-maxctx.sh @b-deploy-1303 $R/flashnext-attn-maxctx-011 0,1,1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for t in 101 110 011; do echo "== attn $t"; grep -h "RESULT\|fails" $R/attn-maxctx-$t.log 2>/dev/null; done
echo ALL_JOBS_DONE
