#!/bin/bash
# 1295 numeric check: same binary, BIGCHERRY_QSA_GATHER=0 (masked path) vs 1 (gathered), no MTP, ~10K and ~80K
# cached context; compares the top-5 probabilities of the first decoded tokens (REPEAT=1 records probs).
# Summation-order noise: same top tokens, probabilities within ~1e-3. Usage: gather-numeric.sh <llama-server> <out>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 REPEAT=1 DECODE_N=16
for depth in 8192 65536; do
  for g in 0 1; do
    BIGCHERRY_QSA_GATHER=$g DEPTH=$depth bash "$s" "$bin" "$root/d$depth/g$g" timing | grep -E "^timing: prompt"
  done
  python3 - "$root/d$depth" "$depth" <<'PY'
import json, sys
r, d = sys.argv[1], sys.argv[2]
a = json.load(open(f"{r}/g0/timing.{d}.probs.json")); b = json.load(open(f"{r}/g1/timing.{d}.probs.json"))
import math
def top(e):  # llama-server: "probs" [{token, prob}] or "top_logprobs" [{token, logprob}]
    ps = e.get("probs") or e.get("top_logprobs") or []
    return [(p["token"], p["prob"] if "prob" in p else math.exp(p["logprob"])) for p in ps]
print(f"d{d}: {len(a)} / {len(b)} tokens with probs")
worst = 0.0
for i, (x, y) in enumerate(zip(a, b)):
    tx, ty = top(x), top(y)
    if not tx or not ty:
        continue
    px, py = dict(tx), dict(ty)
    top_same = tx[0][0] == ty[0][0]
    diff = max(abs(px.get(t, 0) - py.get(t, 0)) for t in set(px) | set(py))
    worst = max(worst, diff)
    print(f"d{d} tok {i}: top1 {'same' if top_same else 'DIFF'} {tx[0][0]!r}/{ty[0][0]!r} max|dp| {diff:.4f}")
print(f"d{d}: worst max|dp| {worst:.4f}")
PY
done
