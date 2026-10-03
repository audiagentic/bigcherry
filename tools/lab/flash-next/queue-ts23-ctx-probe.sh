#!/bin/bash
# Max context for the faster -ts 2.3,2.3,2.4 (−3% ms/step vs 2,2,3 at 128K) with q8_0/q8_0 and f16/q8_0 KV.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1
bash tools/lab/flash-next/fixed-ts-ctx-probe.sh $base /mnt/data/bigcherry-work/runs/flashnext-ts23-ctx 2.3,2.3,2.4 \
  "196608 180224 163840 147456 131072" "q8_0/q8_0 f16/q8_0"
echo ALL_JOBS_DONE
