#!/bin/bash
# QFP17 open question: v6 production QSA mask vs the rebuilt chunked mask (1332) vs an f32 CPU reference at the
# 24K quick-ab prompt (where both 1330 and 1332 diverge from v6 at byte 41). No MTP, 16 greedy tokens, top-5 probs
# per token. Arms: A = v6 (BIGCHERRY_QSA_CHUNK=0), B = chunk 256, C = CPU f32 (no tensor split, FA off).
# Usage: mask-ref.sh <llama-server with 1332> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 REPEAT=1 DECODE_N=16 PROBS_KEEP=16 DEPTH=24576 CTK=f16 CTV=f16
BIGCHERRY_QSA_CHUNK=0   bash "$s" "$bin" "$root/A" timing | grep "^timing: prompt"
BIGCHERRY_QSA_CHUNK=256 bash "$s" "$bin" "$root/B" timing | grep "^timing: prompt"
BIGCHERRY_QSA_CHUNK=0 BIGCHERRY_ATTN_TS= CPU_REF=1 CTX=32768 bash "$s" "$bin" "$root/C" timing | grep "^timing: prompt"
python3 - "$root" <<'PY'
import json, math, sys
r = sys.argv[1]
def load(x):
    out = []
    for e in json.load(open(f"{r}/{x}/timing.24576.probs.json")):
        ps = e.get("probs") or e.get("top_logprobs") or []
        out.append({p["token"]: (p["prob"] if "prob" in p else math.exp(p["logprob"])) for p in ps})
    return out
txt = {x: open(f"{r}/{x}/timing.24576.greedy.txt").read() for x in "ABC"}
for x in "ABC":
    print(x, repr(txt[x][:120]))
A, B, C = load("A"), load("B"), load("C")
for i, (a, b, c) in enumerate(zip(A, B, C)):
    ta, tb, tc = (max(d, key=d.get) if d else None for d in (a, b, c))
    flag = "" if ta == tb == tc else "  <-- differs"
    print(f"tok {i:2d}: A {ta!r} {a.get(ta, 0):.3f} | B {tb!r} {b.get(tb, 0):.3f} | C {tc!r} {c.get(tc, 0):.3f}{flag}")
def dist(x, y):
    return max(max(abs(p.get(t, 0) - q.get(t, 0)) for t in set(p) | set(q)) for p, q in zip(x, y))
print(f"max|dp| A(v6) vs C(f32): {dist(A, C):.4f}   B(chunk) vs C(f32): {dist(B, C):.4f}   A vs B: {dist(A, B):.4f}")
PY
