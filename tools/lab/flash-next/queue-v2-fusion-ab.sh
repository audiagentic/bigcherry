#!/bin/bash
# Adoption ABBA: production profile v2 (b-deploy-1303) vs v2 + 1307 (Q8_1 reuse, cache on) + 1308 (rollback copies
# without CONT) on b-v2-1308, at ~10K and ~80K cached, 512 decode tokens. Same profile flags otherwise.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
BLD=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds  # not B: B is the exported batch size
base=$BLD/6ac162d81fbb92465dc2126e6d157ace/515d3ee6faf4dcccda0ea049f8a5b762/bin/llama-server
new=$BLD/3610525d683306bdc3d70ab830848ab0/b7a406716212f753c39dcc6684c49ba7/bin/llama-server
s=tools/lab/flash-next/long-ctx-profile.sh
R=/mnt/data/bigcherry-work/runs/flashnext-v2-fusion-ab-2
docker stop radiance-vllm >/dev/null 2>&1
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    if [[ $arm == new-* ]]; then bin=$new; envs="GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1"; else bin=$base; envs=""; fi
    echo "== d$depth $arm"
    env $envs DEPTH=$depth bash $s "$bin" "$R/d$depth/$arm" timing | grep -E "^timing:"
  done
done
echo ALL_JOBS_DONE
