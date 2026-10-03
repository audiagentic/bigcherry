#!/bin/bash
# 1303 deep-fill check of the new long-context profile: f16/f16 KV, -c 245760, KV heads pinned to both XTX
# (BIGCHERRY_ATTN_TS=1,1,0 unrotated), expert -ts 0.31,0.27,0.42, + 1302. Fills to DEPTH then decodes; searches
# only filled 8K. Depths: 128K and ~200K.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 B=512 EXTRA_OT='^token_embd\.weight$=CPU'
export BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 TS=0.31,0.27,0.42 CTX=245760 DECODE_N=256
bin=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
R=/mnt/data/bigcherry-work/runs/flashnext-1303-deepfill
docker stop radiance-vllm >/dev/null 2>&1
for d in 131072 204800; do
  echo "== depth $d"
  DEPTH=$d bash tools/lab/flash-next/long-ctx-profile.sh "$bin" "$R/d$d" timing 2>&1 | grep -E "^timing|SERVER_FAILED"
  grep -hE "BIGCHERRY_PATCH_HIT patch=1302|cudaMalloc failed|ALLOC_FAILED" "$R/d$d/timing.server.log" | head -3
  tail -n 4 "$R/d$d/timing.vram.txt" 2>/dev/null
done
echo ALL_JOBS_DONE
