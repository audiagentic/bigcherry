#!/bin/bash
# RNX01 launch census: ~4176 kernel launches per MTP decode step per tensor-split GPU. Are HIP graphs active
# under the meta backend? ABBA decode with GGML_CUDA_DISABLE_GRAPHS=1 vs default at ~10K cached context,
# plus one cache-fill prefill each. No difference => graphs are not in use on this path.
# Usage: graphs-ab.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for arm in on-a off-a off-b on-b; do
  envs=; [[ $arm == off-* ]] && envs="GGML_CUDA_DISABLE_GRAPHS=1"
  echo "== $arm ${envs:-<graphs default>}"
  env $envs DEPTH=8192 bash "$s" "$bin" "$root/$arm" timing
done
