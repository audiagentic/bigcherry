#!/bin/bash
# Adoption ABBA x2 (base new new base base new new base) at ~8K and ~64K cached, DECODE_N tokens per arm, greedy text
# kept per arm. Profile flags come from the caller's environment (both arms); new-arm-only env as trailing args.
# Usage: abba-depths.sh <base llama-server> <new llama-server> <out-root> [NEW_ENV=1 ...]
set -u
base=$1 new=$2 root=$3 newenv="${*:4}"
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b base-c new-c new-d base-d; do
    bin=$base; envs=; [[ $arm == new-* ]] && { bin=$new; envs=$newenv; }
    out=$(env $envs DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
    echo "d$depth $arm: $out"
  done
  t0=$(ls $root/d$depth/base-a/*.greedy.txt 2>/dev/null | head -1)
  for f in $root/d$depth/*/*.greedy.txt; do cmp -s "$t0" "$f" || echo "d$depth DIFFERENT $(dirname $f | xargs basename)"; done
done
