#!/bin/bash
# 1272 host-AllReduce wire sweep on dual XTX (27B Q8_0, -sm tensor, MTP n_max=4).
# Unset GGML_CUDA_AR_WIRE is the pristine path and emits no 1272 marker, so the preflight
# exports an explicit wire (env exported here reaches preflights only; A/B arms set their own).
# Groups: f32 vs bf16 vs f16, then pristine(internal) vs bf16 vs q8_0.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=q8_0
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-ar-wire ar-wire
PREFLIGHT pf-27b-ar-wire-q8 @b-27b-ar-wire $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1272_ar_wire.path=ar_wire_q8_0 $SRV
REQUIRES=pf-27b-ar-wire-q8 AB ab-27b-ar-wire $L/server-ab-ar-wire.json --pairs 6
REQUIRES=pf-27b-ar-wire-q8 AB ab-27b-ar-wire-q8 $L/server-ab-ar-wire-q8.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
