#!/bin/bash
# Decode-path KLD: the prefill-mode KLD (queue-ar-accuracy.sh) only exercises >= 1 MiB reductions,
# which go to RCCL (bf16 >= 32768 elements) for every arm, so it cannot see the host-side wire.
# Here --ubatch-size 1 forces token-sized (~20 KiB) reductions — the decode path adaptive/host
# actually use. Own reference: exact f32 host wire in the same decode mode (8 chunks x 2048 tokens).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
K=tools/lab/ar-accuracy/kld.sh
C=/mnt/data/bigcherry-work/corpus/kld-docs.txt
R=/mnt/data/bigcherry-work/runs/kld-decode-reference.kld
[ -s "$C" ] || { echo "corpus missing; run queue-ar-accuracy.sh first"; exit 3; }
D="--chunks 8 --ubatch-size 1"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD bp-27b-awl bigcherry:stock:linux-multi adaptive-wire-latency gfx1100 bin/llama-perplexity
SCRIPT kldd-ref $K @bp-27b-awl $BC_MODEL $C $R base GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=f32 -- $D
SCRIPT kldd-ref-repeat $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=f32 -- $D
SCRIPT kldd-rccl $K @bp-27b-awl $BC_MODEL $C $R compare -- $D --allreduce ccl
SCRIPT kldd-host-bf16 $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal -- $D
SCRIPT kldd-host-f16 $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=f16 -- $D
SCRIPT kldd-adaptive-f32 $K @bp-27b-awl $BC_MODEL $C $R compare GGML_CUDA_AR_WIRE=f32 -- $D --allreduce adaptive
SCRIPT kldd-adaptive-bf16 $K @bp-27b-awl $BC_MODEL $C $R compare -- $D --allreduce adaptive
JOBS
status=0
bash tools/lab/plan-qualification/queue.sh "$jobs" || status=1
echo "QUEUE_EXIT=$status $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py kld /mnt/data/bigcherry-work/runs kldd-ref-repeat kldd-rccl kldd-host-bf16 kldd-host-f16 kldd-adaptive-f32 kldd-adaptive-bf16 || status=1
rm -f "$jobs"
echo ALL_JOBS_DONE
exit "$status"
