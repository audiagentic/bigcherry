#!/bin/bash
# QFP17 mask correctness, second pass: (1) CPU f32 reference with room for the 38.7K-token "24K" prompt;
# (2) 1332 chunked, no MTP, under gdb - it segfaulted in libggml-base at the first single-token decode.
# Compares against A from mask-ref.sh (v6 GPU, already in <root>/A). Usage: mask-ref2.sh <llama-server with 1332> <root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 REPEAT=1 DECODE_N=16 PROBS_KEEP=16 DEPTH=24576 CTK=f16 CTV=f16
BIGCHERRY_QSA_CHUNK=0 BIGCHERRY_ATTN_TS= CPU_REF=1 CTX=40960 bash "$s" "$bin" "$root/C" timing | grep "^timing: prompt"
BIGCHERRY_QSA_CHUNK=256 WRAP="gdb -batch -ex run -ex bt -ex quit --args" bash "$s" "$bin" "$root/Bgdb" timing | grep "^timing: prompt"
grep -A25 "SIGSEGV" "$root/Bgdb/timing.server.log" | head -30
python3 - "$root" <<'PY'
import json, math, sys
r = sys.argv[1]
def load(x):
    out = []
    for e in json.load(open(f"{r}/{x}/timing.24576.probs.json")):
        ps = e.get("probs") or e.get("top_logprobs") or []
        out.append({p["token"]: (p["prob"] if "prob" in p else math.exp(p["logprob"])) for p in ps})
    return out
for x in "AC":
    print(x, repr(open(f"{r}/{x}/timing.24576.greedy.txt").read()[:120]))
A, C = load("A"), load("C")
for i, (a, c) in enumerate(zip(A, C)):
    ta, tc = (max(d, key=d.get) if d else None for d in (a, c))
    print(f"tok {i:2d}: A(v6) {ta!r} {a.get(ta, 0):.3f} | C(f32) {tc!r} {c.get(tc, 0):.3f}{'' if ta == tc else '  <-- differs'}")
print("max|dp| A(v6) vs C(f32):", round(max(max(abs(p.get(t, 0) - q.get(t, 0)) for t in set(p) | set(q)) for p, q in zip(A, C)), 4))
PY
