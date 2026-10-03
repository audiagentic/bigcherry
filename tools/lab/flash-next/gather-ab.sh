#!/bin/bash
# 1295 (QSA gathered-cell decode attention) A/B against 1291+1292+1294, f16 draft KV, 192K deployment config:
# ABBA decode at ~10K / ~80K / ~160K cached context (MTP3), then no-MTP greedy at 10K and 80K (new vs base).
# Usage: gather-ab.sh <base llama-server> <new llama-server> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export CTKD=f16 CTVD=f16 BIGCHERRY_PATCH_TRACE=1
for depth in 8192 65536 131072; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
done
grep -h "patch=1295" "$root"/d8192/new-a/*.log | head -1
for depth in 8192 65536; do
  for arm in base new; do
    bin=$base; [ $arm = new ] && bin=$new
    NO_MTP=1 DEPTH=$depth bash "$s" "$bin" "$root/parity-d$depth/$arm" timing
  done
  python3 - "$root/parity-d$depth" "$depth" <<'PY'
import sys
r, d = sys.argv[1], sys.argv[2]
a = open(f"{r}/base/timing.{d}.greedy.txt").read(); b = open(f"{r}/new/timing.{d}.greedy.txt").read()
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(f"parity no-MTP d{d}: common prefix {n}/{min(len(a), len(b))}{' (identical)' if a == b else ''}")
PY
done
