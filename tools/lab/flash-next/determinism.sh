#!/bin/bash
# Greedy non-determinism at long context (no MTP): two runs of one binary diverged at the first token at
# ~80K but matched at ~10K. Per (allreduce provider, depth): two server starts, each filling the cache once
# and decoding twice on the cached prefix, recording greedy text and top-5 probs of the first tokens.
#   same server r0 vs r1 differ  -> decode-side non-determinism
#   r0 matches within a server but differs across starts -> prefill/cache-fill non-determinism
# Usage: determinism.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 REPEAT=1 DECODE_N=64
for ar in cpu-root ccl; do
  for depth in 32768 65536; do
    for start in s1 s2; do
      AR=$ar DEPTH=$depth bash "$s" "$bin" "$root/$ar/d$depth/$start" timing
    done
    python3 - "$root/$ar/d$depth" "$depth" <<'PY'
import json, sys
root, d = sys.argv[1], sys.argv[2]
def txt(p): return open(p).read()
def pre(a, b): return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
def top(p):
    try: return [(c["token"], round(c["probs"][0]["prob"], 6)) for c in json.load(open(p))[:3]]
    except Exception: return []
for s in ("s1", "s2"):
    a, b = txt(f"{root}/{s}/timing.{d}.greedy.txt"), txt(f"{root}/{s}/timing.{d}.r1.greedy.txt")
    print(f"{root} {s}: within-server r0 vs r1 common {pre(a, b)}/{min(len(a), len(b))}")
a, b = txt(f"{root}/s1/timing.{d}.greedy.txt"), txt(f"{root}/s2/timing.{d}.greedy.txt")
print(f"{root}: across starts r0 common {pre(a, b)}/{min(len(a), len(b))}")
print("  s1 first tokens", top(f"{root}/s1/timing.{d}.probs.json"))
print("  s2 first tokens", top(f"{root}/s2/timing.{d}.probs.json"))
PY
  done
done
