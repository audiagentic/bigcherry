#!/bin/bash
# 1273 IQ MMVQ tuning on one 7900 XTX (gfx1100): pristine vs VDR vs VDR+NWARPS arms on one binary,
# for unsloth Qwen3.8-27B UD-IQ4_XS and UD-IQ3_XXS. Preflights prove the tuned path fires
# (env exported here only reaches preflights; A/B arms set their own env).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf
export BIGCHERRY_IQ_MMVQ_VDR=1
M=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-
D=tools/lab/quant-sweep
SRV="-ngl 99 --fit off -c 8192 --flash-attn on"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-iq-mmvq iq-mmvq gfx1100,gfx1201
VIS=0 PREFLIGHT pf-iq-mmvq-iq4 @b-iq-mmvq ${M}UD-IQ4_XS.gguf BIGCHERRY_PATCH_HIT.patch=1273_iq_mmvq.*type=iq4_xs.*arch=gfx1100 $SRV
VIS=0 PREFLIGHT pf-iq-mmvq-iq3 @b-iq-mmvq ${M}UD-IQ3_XXS.gguf BIGCHERRY_PATCH_HIT.patch=1273_iq_mmvq.*type=iq3_xxs.*arch=gfx1100 $SRV
VIS=0 REQUIRES=pf-iq-mmvq-iq4 AB ab-iq-mmvq-xtx1-iq4 $D/iq-mmvq-xtx1-iq4_xs.json --pairs 6
VIS=0 REQUIRES=pf-iq-mmvq-iq3 AB ab-iq-mmvq-xtx1-iq3 $D/iq-mmvq-xtx1-iq3_xxs.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
