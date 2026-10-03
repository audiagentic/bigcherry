#!/bin/bash
# QFP07/QFP09 follow-ups on production profile v2 (build b-deploy-1303):
# (1) f16/f16 max-context search with attention/KV rotated over all three GPUs (BIGCHERRY_ATTN_TS=1,1,1 rotate=1)
#     while the expert -ts adapts -- the speed-oriented placement for long-context decode;
# (2) synctrace of decode at ~80K on profile v2 -- attribute the ~58% idle gaps to hipStreamSynchronize call sites.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
bin=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
R=/mnt/data/bigcherry-work/runs
docker stop radiance-vllm >/dev/null 2>&1
echo "== rotated attention f16/f16 max ctx"
bash tools/lab/flash-next/attn-maxctx.sh "$bin" "$R/flashnext-attn-maxctx-111r-f16" 1,1,1 1 131072 245760 f16 f16
echo "== synctrace ~80K profile v2"
(
  export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
  export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
  export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
  DEPTH=65536 bash tools/lab/flash-next/long-ctx-profile.sh "$bin" "$R/flashnext-v2-synctrace" synctrace 2>&1 | tail -n 60
)
echo ALL_JOBS_DONE
