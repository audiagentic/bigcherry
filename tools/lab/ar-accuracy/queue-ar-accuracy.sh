#!/bin/bash
# Accuracy gates for lossy AllReduce wires on Qwen3.8-27B Q8_0, dual 7900 XTX, -sm tensor.
# Reference: host pipeline with an explicit f32 wire (1272). Pristine RCCL is NOT exact: it
# switches 2-GPU reductions to bf16 at >= 32768 elements. Compared: host pipeline pristine (bf16 by default), 1272
# host f32 / f16 / q8_0, and 0840 adaptive. Gates: tools/lab/ar-accuracy/gates.py.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
K=tools/lab/ar-accuracy/kld.sh
C=/mnt/vault/development/llmhosts/llamacpp/bench/corpus/ppl-default.txt
R=/mnt/data/bigcherry-work/runs/kld-27b-reference.kld
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD bp-27b-ar-wire ar-wire gfx1100 bin/llama-perplexity
BUILD bp-27b-adaptive allreduce-adaptive gfx1100 bin/llama-perplexity
SCRIPT kld-27b-ref $K @bp-27b-ar-wire $BC_MODEL $C $R base GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=f32
SCRIPT kld-27b-host-pristine $K @bp-27b-ar-wire $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal
SCRIPT kld-27b-rccl $K @bp-27b-ar-wire $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=nccl
SCRIPT kld-27b-host-f16 $K @bp-27b-ar-wire $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=f16
SCRIPT kld-27b-host-q8 $K @bp-27b-ar-wire $BC_MODEL $C $R compare GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=q8_0
SCRIPT kld-27b-adaptive $K @bp-27b-adaptive $BC_MODEL $C $R compare -- --allreduce adaptive
JOBS
status=0
bash tools/lab/plan-qualification/queue.sh "$jobs" || status=1
echo "QUEUE_EXIT=$status $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py kld /mnt/data/bigcherry-work/runs kld-27b-host-pristine kld-27b-rccl kld-27b-host-f16 kld-27b-host-q8 kld-27b-adaptive || status=1
python3 tools/lab/ar-accuracy/gates.py acceptance /mnt/data/bigcherry-work/runs/ab-27b-allreduce/result /mnt/data/bigcherry-work/runs/ab-27b-ar-wire/result /mnt/data/bigcherry-work/runs/ab-27b-ar-wire-q8/result /mnt/data/bigcherry-work/runs/ab-27b-adaptive/result || status=1
rm -f "$jobs"
echo ALL_JOBS_DONE
exit "$status"
