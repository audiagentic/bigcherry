#!/bin/bash
# Deployment-profile ABBA (a whole-profile comparison, not a single-variable one): base = current production
# (b-flash-deploy-1: q8_0/q8_0 KV, -c 196608, -ts 2,2,3); new = 1303 long-context profile (deploy + 1302 + 1303:
# f16/f16 KV, -c 245760, BIGCHERRY_ATTN_TS=1,1,0 unrotated, expert -ts 0.31,0.27,0.42, -b 512, token_embd on CPU).
# Shared: Q5_K_M MTP3 draft on the 6900, f16 draft KV, 64K draft vocab, ub512, cpu-root. ~10K and ~80K cached.
# Usage: profile-ab.sh <base llama-server> <new llama-server> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTKD=f16 CTVD=f16 DECODE_N=512
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    if [[ $arm == new-* ]]; then
      envs="CTK=f16 CTV=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 EXTRA_OT=^token_embd\.weight\$=CPU BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0"
      bin=$new
    else
      envs="CTK=q8_0 CTV=q8_0 CTX=196608 TS=2,2,3"
      bin=$base
    fi
    echo "== d$depth $arm"
    env $envs DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing | grep -E "^timing:"
  done
done
