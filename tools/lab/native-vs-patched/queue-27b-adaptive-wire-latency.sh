#!/bin/bash
# Promotion-candidate matrix on one binary (0860+1225+0840+1272+1275), dual XTX 27B Q8_0 MTP,
# --allreduce adaptive: wire f32 / bf16 / f16 for adaptive's host side, each with and without
# 1275's latency switches (BIGCHERRY_AR_SLOT_SYNC=none, BIGCHERRY_AR_SMALL_BLOCKS=1).
# 1275 was measured neutral (slot_sync=none) / -8.5% decode (small_blocks=1), so its arms were
# dropped: one group adaptive f32 vs bf16 vs f16. Question: does f16 give bf16 speed at RCCL-level
# MTP acceptance? Acceptance per arm via tools/lab/ar-accuracy/gates.py acceptance.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export GGML_CUDA_AR_WIRE=f32 BIGCHERRY_AR_SLOT_SYNC=none BIGCHERRY_AR_SMALL_BLOCKS=1
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4 --allreduce adaptive"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-awl adaptive-wire-latency
PREFLIGHT pf-27b-awl @b-27b-awl $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1275_ar_small.*slot_sync=none $SRV
REQUIRES=pf-27b-awl AB ab-27b-awl-1 $L/server-ab-awl-1.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
R=/mnt/data/bigcherry-work/runs
python3 tools/lab/ar-accuracy/gates.py acceptance $R/ab-27b-awl-1/result
rm -f "$jobs"
echo ALL_JOBS_DONE
