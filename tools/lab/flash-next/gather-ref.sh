#!/bin/bash
# 1295 accuracy vs an f32 reference: A = masked path FA on, B = gathered path FA on, C = masked path FA off
# (mul_mat + f32 softmax). Same binary, f16 KV, no MTP, ~10K cached prompt; top-5 probs of 8 decoded tokens.
# Usage: gather-ref.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 REPEAT=1 DECODE_N=16 CTK=f16 CTV=f16 DEPTH=8192 CTX=${CTX:-65536}
BIGCHERRY_QSA_GATHER=0 FA=on  bash "$s" "$bin" "$root/A" timing | grep "^timing: prompt"
BIGCHERRY_QSA_GATHER=1 FA=on  bash "$s" "$bin" "$root/B" timing | grep "^timing: prompt"
BIGCHERRY_QSA_GATHER=0 FA=off bash "$s" "$bin" "$root/C" timing | grep "^timing: prompt"
python3 - "$root" <<'PY'
import json, math, sys
r = sys.argv[1]
def load(x):
    out = []
    for e in json.load(open(f"{r}/{x}/timing.8192.probs.json")):
        ps = e.get("probs") or e.get("top_logprobs") or []
        out.append({p["token"]: (p["prob"] if "prob" in p else math.exp(p["logprob"])) for p in ps})
    return out
A, B, C = load("A"), load("B"), load("C")
def dist(x, y):
    return max(max(abs(a.get(t, 0) - b.get(t, 0)) for t in set(a) | set(b)) for a, b in zip(x, y))
print(f"max|dp| masked-FA vs f32 ref: {dist(A, C):.4f}")
print(f"max|dp| gather-FA vs f32 ref: {dist(B, C):.4f}")
print(f"max|dp| masked-FA vs gather-FA: {dist(A, B):.4f}")
PY
