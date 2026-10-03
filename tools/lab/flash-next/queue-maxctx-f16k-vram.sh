#!/bin/bash
# Max f16-K/q8_0-V context with every GPU filled: adaptive split search (maxctx-search.py) plus the zero-code QFN03
# reserves -- token_embd on CPU and -b equal to -ub (512). Previous best without them: 144K ub512 / 160K ub384.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTKD=f16 CTVD=f16 B=512 EXTRA_OT='^token_embd\.weight$=CPU'
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1
python3 tools/lab/flash-next/maxctx-search.py $base /mnt/data/bigcherry-work/runs/flashnext-maxctx-f16k-vram f16 q8_0 512 147456 196608
echo ALL_JOBS_DONE
