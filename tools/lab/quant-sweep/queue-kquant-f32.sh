#!/bin/bash
# 1274 Q6_K/Q4_K F32-activation decode vs the production build (b-27b-control, which has 1241),
# unsloth Qwen3.8-27B UD-Q6_K and UD-Q4_K_M, one XTX and both XTXs (-sm tensor). MTP off.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
M=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-
export BC_MODEL=${M}UD-Q6_K.gguf
D=tools/lab/quant-sweep
SRV="-ngl 99 --fit off -c 8192 --flash-attn on"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-kquant-f32 kquant-f32
VIS=0 PREFLIGHT pf-kquant-q6 @b-kquant-f32 ${M}UD-Q6_K.gguf BIGCHERRY_PATCH_HIT.patch=1274_kquant_f32.*type=q6_k $SRV
VIS=0 PREFLIGHT pf-kquant-q4 @b-kquant-f32 ${M}UD-Q4_K_M.gguf BIGCHERRY_PATCH_HIT.patch=1274_kquant_f32.*type=q4_k $SRV
VIS=0 REQUIRES=pf-kquant-q6 AB ab-kquant-xtx1-q6 $D/kquant-f32-xtx1-q6_k.json --pairs 4
VIS=0 REQUIRES=pf-kquant-q4 AB ab-kquant-xtx1-q4 $D/kquant-f32-xtx1-q4_k_m.json --pairs 4
VIS=0,1 REQUIRES=pf-kquant-q6 AB ab-kquant-xtx2-q6 $D/kquant-f32-xtx2-q6_k.json --pairs 4
VIS=0,1 REQUIRES=pf-kquant-q4 AB ab-kquant-xtx2-q4 $D/kquant-f32-xtx2-q4_k_m.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
