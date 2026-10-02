#!/bin/bash
# P2P copy-correctness gate (1252), dual XTX 27B Q8_0, P2P-enabled kernel only.
# Reference: host staging with an explicit f32 wire. With the same f32 wire the P2P path performs
# the same arithmetic, so its logits must match the reference exactly (mean KLD ~ 0, same top
# 100%); any KLD above noise means a corrupted peer copy. bf16/q8_0 over P2P are then measured
# against the same reference like the other lossy wires. 32 x 2048-token chunks of prefill, so
# the copy-engine/P2P path carries every reduction >= 1 MiB.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
K=tools/lab/ar-accuracy/kld.sh
C=/mnt/vault/development/llmhosts/llamacpp/bench/corpus/ppl-default.txt
R=/mnt/data/bigcherry-work/runs/kld-27b-p2p-reference.kld
H="GGML_CUDA_ALLREDUCE=internal"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
BUILD bp-27b-p2p bigcherry:stock:linux-multi ar-p2p gfx1100 bin/llama-perplexity
SCRIPT kld-p2p-ref $K @bp-27b-p2p $BC_MODEL $C $R base $H GGML_CUDA_AR_WIRE=f32
SCRIPT kld-p2p-f32 $K @bp-27b-p2p $BC_MODEL $C $R compare $H GGML_CUDA_AR_WIRE=f32 GGML_CUDA_AR_P2P=1
SCRIPT kld-p2p-f32-repeat $K @bp-27b-p2p $BC_MODEL $C $R compare $H GGML_CUDA_AR_WIRE=f32 GGML_CUDA_AR_P2P=1
SCRIPT kld-p2p-bf16 $K @bp-27b-p2p $BC_MODEL $C $R compare $H GGML_CUDA_AR_P2P=1
SCRIPT kld-p2p-q8 $K @bp-27b-p2p $BC_MODEL $C $R compare $H GGML_CUDA_AR_WIRE=q8_0 GGML_CUDA_AR_P2P=1
JOBS
status=0
bash tools/lab/plan-qualification/queue.sh "$jobs" || status=1
echo "QUEUE_EXIT=$status $(date -Is)"
R2=/mnt/data/bigcherry-work/runs
for r in kld-p2p-f32 kld-p2p-f32-repeat; do
  grep -aE "Mean +KLD|Same top p|Maximum KLD" $R2/$r/perplexity.log 2>/dev/null | sed "s/^/$r: /"
done
python3 tools/lab/ar-accuracy/gates.py kld $R2 kld-p2p-f32 kld-p2p-f32-repeat kld-p2p-bf16 kld-p2p-q8 || status=1
rm -f "$jobs"
echo ALL_JOBS_DONE
exit "$status"
