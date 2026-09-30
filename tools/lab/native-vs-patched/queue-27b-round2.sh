#!/bin/bash
# Round 2 dual-XTX 27B queue: 0860 --allreduce CLI provider matrix (preflights prove each
# provider is selected, then a 3-arm A/B on one binary) and 1255/1268 adaptive MTP depth
# (same binary; the adaptive arm only adds --spec-draft-n-min-adaptive; preflight requires a
# real depth change).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
P=BIGCHERRY_PATCH_HIT.patch=0860_allreduce_provider_cli
# Adaptive MTP only changes depth after several verify steps: generate long enough to see it.
export BC_PREFLIGHT_N_PREDICT=256
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-0860 allreduce-cli
BUILD b-27b-adaptive-mtp adaptive-mtp
PREFLIGHT pf-0860-auto @b-27b-0860 $BC_MODEL $P.provider=ccl.wire=native $SRV --allreduce auto
PREFLIGHT pf-0860-ccl @b-27b-0860 $BC_MODEL $P.provider=ccl.wire=native $SRV --allreduce ccl
PREFLIGHT pf-0860-host @b-27b-0860 $BC_MODEL $P.provider=host.wire=native $SRV --allreduce host
PREFLIGHT pf-0860-butterfly @b-27b-0860 $BC_MODEL $P.provider=butterfly.wire=native $SRV --allreduce butterfly
REQUIRES=pf-0860-butterfly AB ab-27b-0860-providers $L/server-ab-0860-providers.json --pairs 6
PREFLIGHT pf-27b-adaptive-mtp @b-27b-adaptive-mtp $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1268_prbe52_adaptive_mtp_wiring.*event=depth_change $SRV --spec-draft-n-min-adaptive 2
REQUIRES=pf-27b-adaptive-mtp AB ab-27b-adaptive-mtp $L/server-ab-adaptive-mtp.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
