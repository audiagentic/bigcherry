#!/bin/bash
# 1295 QSA gather decode on production profile v2 (f16/f16 240K, KV on both XTX): quick ABA screens at ~80K and
# ~160K cached (gain grew with context on the old profile: -3.7% @80K, -12% @160K). Base = b-deploy-1303.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v2-1295 bigcherry:stock:linux-multi deploy-v2-plus-1295 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1295-d65536 tools/lab/flash-next/quick-ab-depth.sh 65536 $base @b-v2-1295 $R/flashnext-v2-1295-d65536
VIS=0,1,2,3 SCRIPT v2-1295-d131072 tools/lab/flash-next/quick-ab-depth.sh 131072 $base @b-v2-1295 $R/flashnext-v2-1295-d131072
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in 65536 131072; do echo "== d$d"; tail -n 4 $R/v2-1295-d$d.log 2>/dev/null; done
echo ALL_JOBS_DONE
