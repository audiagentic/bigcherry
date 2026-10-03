#!/bin/bash
# Draft model quantization A/B (6900 decode is bandwidth-bound; the MTP draft is Q8_0). Builds llama-quantize in
# the given build tree, requantizes the qsa4 MTP sidecar to Q5_K_M and Q4_K_M (--allow-requantize: no BF16
# source exists), then ABBA decode at ~10K and ~80K (MTP3, f16 draft KV) Q8_0 vs Q5_K_M vs Q4_K_M.
# Output cannot change (the target verifies); watch ms/step and acceptance. Usage: draft-quant.sh <llama-server> <out>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
bdir=$(dirname "$(dirname "$bin")")
mtp=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP
src=$mtp/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf
cmake --build "$bdir" --target llama-quantize -j 16 > "$root.quantize-build.log" 2>&1 || { echo "llama-quantize build failed"; tail -5 "$root.quantize-build.log"; exit 1; }
for q in Q5_K_M Q4_K_M; do
  dst=$mtp/mtp-Qwen3.8-Flash-Next-$q-qsa4.gguf
  [ -f "$dst" ] || "$bdir/bin/llama-quantize" --allow-requantize "$src" "$dst" "$q" > "$root.quantize-$q.log" 2>&1 || { echo "quantize $q failed"; tail -3 "$root.quantize-$q.log"; exit 1; }
  ls -la "$dst"
done
export CTKD=f16 CTVD=f16
for depth in 8192 65536; do
  for arm in q8-a q5-a q4-a q4-b q5-b q8-b; do
    case ${arm%-*} in q8) d=$src;; q5) d=$mtp/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf;; q4) d=$mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M-qsa4.gguf;; esac
    echo "== d$depth $arm"
    DRAFT=$d DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing | grep -E "^timing: prompt"
  done
done
