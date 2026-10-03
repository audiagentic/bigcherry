#!/bin/bash
# 1295 v2 full ABBA on the deployment candidate (deployment draft env in both arms): ~10K (below the 32K gather
# threshold: should be identical), ~80K and ~160K cached context. Usage: gather-v2-ab.sh <base> <new> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export CTKD=f16 CTVD=f16 BIGCHERRY_DRAFT_VOCAB_N=65536
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
for depth in 8192 65536 131072; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    echo "== d$depth $arm"
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing | grep -E "^timing:"
  done
done
