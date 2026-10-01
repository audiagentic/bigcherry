#!/bin/bash
# Rerun + uneven -ts sweep (1:1:1, 29:29:22, 3:3:2; RCCL; 64-row split rounding makes 3:3:2 exact). 3-GPU (2x 7900 XTX + R9700) Qwen3.8-27B BF16 (~55 GB, needs all three cards): RCCL vs 1276
# adaptive N=3 (root3 below the 1 MiB switch, RCCL above) (host arm dropped: it segfaulted on the first request), one
# multi-arch binary, no MTP. Needs the R9700 free: stops radiance-vllm and restarts it after.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf
L=tools/lab/native-vs-patched
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2 BUILD b-3g-nway ar-adaptive-nway gfx1100,gfx1201
VIS=0,1,2 AB ab-3g-bf16-nway-2 $L/server-ab-3g-bf16-nway-2.json --pairs 6
VIS=0,1,2 AB ab-3g-bf16-ts $L/server-ab-3g-bf16-ts.json --pairs 6
VIS=0,1,2 SCRIPT host3-crash-1 $L/host3-crash-repro.sh @b-3g-nway /mnt/data/bigcherry-work/runs/host3-crash-1/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
