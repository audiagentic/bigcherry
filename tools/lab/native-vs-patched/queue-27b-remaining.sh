#!/bin/bash
# Remaining dual-XTX 27B tests (tierL-qwen27b-q8, GPUs 0,1, -sm tensor, MTP n_max=4) through the queue:
#   AllReduce provider matrix (ccl/host/none via GGML_CUDA_ALLREDUCE on one control binary),
#   0840 adaptive vs control, and 1205 (rd12) / 1261 (nro10) gated on firing preflights.
# Every row holds the host-exclusive + GPU 0,1 locks; finished rows are skipped on re-run.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
SRV="-sm tensor -ngl 99 --fit off -c 64000 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 4"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD b-27b-control -
BUILD b-27b-adaptive allreduce-adaptive
BUILD b-27b-rd12 rd12-only
BUILD b-27b-nro10 nro10-only
AB ab-27b-allreduce $L/server-ab-allreduce.json --pairs 6
AB ab-27b-adaptive $L/server-ab-adaptive.json --pairs 6
PREFLIGHT pf-27b-rd12 @b-27b-rd12 $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1205_rd12 $SRV
PREFLIGHT pf-27b-nro10 @b-27b-nro10 $BC_MODEL BIGCHERRY_PATCH_HIT.patch=1261_nro10 $SRV
REQUIRES=pf-27b-rd12 AB ab-27b-rd12 $L/server-ab-rd12.json --pairs 4
REQUIRES=pf-27b-nro10 AB ab-27b-nro10 $L/server-ab-nro10.json --pairs 4
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
