#!/bin/bash
# Round 3 dual-XTX 27B queue: 1263 (PRBE41) declines channels-major under a tensor split. The preflight must
# show the fallback marker under -sm tensor (the channels-major kernel still runs, replicated)
# before the A/B against the plain control build (b-27b-control from queue-27b-remaining.sh).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-1263 bigcherry:stock:linux-multi ssm-conv-channels-major gfx1100
PREFLIGHT pf-27b-1263-split @b-27b-1263 $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1263_prbe41.path=ssm_conv_channels_major_declined_split $SRV
REQUIRES=pf-27b-1263-split AB ab-27b-1263 $L/server-ab-1263.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
