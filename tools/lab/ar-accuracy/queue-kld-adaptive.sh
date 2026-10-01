#!/bin/bash
# KLD of the promotion candidates (adaptive closure + 1272 + 1275, --allreduce adaptive) against the
# host-f32 reference produced by queue-ar-accuracy.sh (same frozen corpus, same reference file).
# Arms: adaptive with exact f32 wire (expected ~= reference), bf16 (pristine policy), f16, and
# f32 + 1275 switches (must equal f32: 1275 changes scheduling, not arithmetic).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
K=tools/lab/ar-accuracy/kld.sh
C=/mnt/data/bigcherry-work/corpus/kld-docs.txt
R=/mnt/data/bigcherry-work/runs/kld-27b-reference.kld
[ -s "$R" ] && [ -s "$C" ] || { echo "reference or corpus missing; run queue-ar-accuracy.sh first"; exit 3; }
A="-- --allreduce adaptive"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD bp-27b-awl adaptive-wire-latency gfx1100 bin/llama-perplexity
SCRIPT kld-awl-f32 $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_AR_WIRE=f32 $A
SCRIPT kld-awl-f32-fast $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_AR_WIRE=f32 BIGCHERRY_AR_SLOT_SYNC=none BIGCHERRY_AR_SMALL_BLOCKS=1 $A
SCRIPT kld-awl-bf16 $K @bp-27b-awl $BC_MODEL $C $R compare $A
SCRIPT kld-awl-f16 $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_AR_WIRE=f16 $A
JOBS
status=0
bash tools/lab/plan-qualification/queue.sh "$jobs" || status=1
echo "QUEUE_EXIT=$status $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py kld /mnt/data/bigcherry-work/runs kld-awl-f32 kld-awl-f32-fast kld-awl-bf16 kld-awl-f16 || status=1
rm -f "$jobs"
echo ALL_JOBS_DONE
exit "$status"
