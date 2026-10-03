#!/bin/bash
# 1297 trimmed MTP draft head: one binary, BIGCHERRY_DRAFT_VOCAB_N unset vs 16384 / 32768 / 65536, decode at
# ~10K and ~80K cached context (MTP3, f16 draft KV, 192K deployment config). Output cannot change (target
# verifies); watch ms/step and acceptance. Usage: trim-ab.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export CTKD=f16 CTVD=f16
for depth in 8192 65536; do
  for arm in full-a n64k-a n64k-b full-b; do
    n=${arm%-*}; n=${n#n}
    case $n in full) envs=;; 16k) envs=BIGCHERRY_DRAFT_VOCAB_N=16384;; 32k) envs=BIGCHERRY_DRAFT_VOCAB_N=32768;; 64k) envs=BIGCHERRY_DRAFT_VOCAB_N=65536;; esac
    echo "== d$depth $arm ${envs:-<full vocab>}"
    env $envs DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing | grep -E "^timing: prompt"
    grep -h "BigCherry 1297" "$root/d$depth/$arm/"*.log 2>/dev/null | head -1
  done
done
