#!/bin/bash
# P2P AllReduce on dual XTX (only while Brutus runs the P2P-enabled kernel), 27B Q8_0 MTP.
# Group 1: RCCL vs host staging vs host + P2P (1252). Group 2: P2P with bf16 (pristine) vs f32
# vs q8_0 wire (1272/1250). The preflight requires 1252's marker, i.e. the content probe passed
# and the P2P path really executed (it falls back to host staging on any mismatch).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_P2P=1
# 1252's P2P path only runs on copy-engine (>= 1 MiB) reductions, i.e. prefill: use a long prompt.
export BC_PREFLIGHT_PROMPT_REPEAT=200
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-p2p ar-p2p
PREFLIGHT pf-27b-p2p @b-27b-p2p $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1252_nro03 $SRV
REQUIRES=pf-27b-p2p AB ab-27b-p2p $L/server-ab-p2p.json --pairs 6
REQUIRES=pf-27b-p2p AB ab-27b-p2p-wire $L/server-ab-p2p-wire.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
