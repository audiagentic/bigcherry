#!/bin/bash
# QFP06 / 1304 on production profile v2: (1) BIGCHERRY_GRAPH_MEMLOG=1 at ~80K to measure per-instance graph memory
# and how many graphs a server accumulates; (2) quick ABA at ~24K and ~80K: base = no cap, new = cap 32 (decode cost
# of eviction/recapture churn). Same binary in both arms.
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
VIS=0,1,2,3 BUILD b-v2-1304 bigcherry:stock:linux-multi deploy-v2-plus-1304 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT v2-1304-memlog tools/lab/flash-next/graph-memlog.sh @b-v2-1304 $R/flashnext-v2-1304-memlog
VIS=0,1,2,3 SCRIPT v2-1304-cap32-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-v2-1304 @b-v2-1304 $R/flashnext-v2-1304-cap32-d24k BIGCHERRY_CUDA_GRAPH_CAP=32
VIS=0,1,2,3 SCRIPT v2-1304-cap32-d80k tools/lab/flash-next/quick-ab-depth.sh 65536 @b-v2-1304 @b-v2-1304 $R/flashnext-v2-1304-cap32-d80k BIGCHERRY_CUDA_GRAPH_CAP=32
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for j in v2-1304-memlog v2-1304-cap32-d24k v2-1304-cap32-d80k; do echo "== $j"; tail -n 12 $R/$j.log 2>/dev/null; done
echo ALL_JOBS_DONE
