#!/bin/bash
# Deep decode profile of production profile v2 (QFP09): rocprofv3 kernel trace of the decode window at ~10K, ~80K
# and ~200K cached tokens, then a per-GPU op-class census (rank-census.py) and the per-kernel table.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 DECODE_N=256
bin=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
R=/mnt/data/bigcherry-work/runs/flashnext-v2-profile
docker stop radiance-vllm >/dev/null 2>&1
for d in 8192 65536 163840; do
  echo "== depth $d"
  DEPTH=$d bash tools/lab/flash-next/long-ctx-profile.sh "$bin" "$R/d$d" decode 2>&1 | tail -n 30
  python3 tools/lab/flash-next/rank-census.py "$R/d$d"
done
echo ALL_JOBS_DONE
