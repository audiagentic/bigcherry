#!/bin/bash
# ABBA: production 192K q8_0 profile vs the 1303 f16/f16 240K profile, at ~10K and ~80K cached context.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
B=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds
base=$B/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
new=$B/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1
bash tools/lab/flash-next/profile-ab.sh "$base" "$new" /mnt/data/bigcherry-work/runs/flashnext-profile-ab-1
echo ALL_JOBS_DONE
