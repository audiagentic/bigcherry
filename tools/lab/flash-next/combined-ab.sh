#!/bin/bash
# End-to-end check that the separately measured wins add up. Base = this morning's deployment (1291 only, Q8_0
# draft, q8_0 draft KV, full draft vocab); new = 1291+1292+1294+1297 + Q5_K_M draft + f16 draft KV + 64K draft
# vocab. Same 192K deployment flags otherwise (-ts 2,2,3, q8_0 target KV, ub512, MTP3, cpu-root). ABBA at ~10K and
# ~80K cached context. Usage: combined-ab.sh <base llama-server> <new llama-server> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
mtp=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    if [[ $arm == new-* ]]; then
      envs="CTKD=f16 CTVD=f16 DRAFT=$mtp/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf BIGCHERRY_DRAFT_VOCAB_N=65536"; bin=$new
    else
      envs="CTKD=q8_0 CTVD=q8_0 DRAFT=$mtp/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf"; bin=$base
    fi
    echo "== d$depth $arm"
    env $envs DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing | grep -E "^timing:"
  done
done
