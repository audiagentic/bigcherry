#!/bin/bash
# Quick direction check before a full on/off comparison: one helper-on load of Flash-Next.
# PASS needs helper calls on both ranks, decode above the helper-off reference and prefill
# within PREFILL_TOL of it. References default to the 2026-10-13 helper-off runs (1,224-1,239
# prefill, 26.4-26.6 decode on the 7,306-token prompt).
# Usage: B=<lane build dir> tools/lab/radiance/fn-helper-gate.sh <out-dir>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?out dir}
B=${B:?set B to the lane build directory}
ref_prefill=${REF_PREFILL:-1230}
ref_decode=${REF_DECODE:-26.5}
tol=${PREFILL_TOL:-0.05}

rm -f /dev/shm/r3-moe-helper
HELPER=1 EXTRA_ARGS="${EXTRA_ARGS:---gpu-headroom-mib 320}" RAD_BIN="$B/bin/radiance" RADIANCE_HOME="$B/radiance_home" \
    timeout 2400 bash "$here/fn-helper.sh" "$out" >"$out.run" 2>&1
grep -E "prefill|decode|EXIT|TIMEOUT" "$out.run"
grep -E "R3_MOE_HELPER rank" "$out/server.log"

python3 - "$out.run" "$out/server.log" "$ref_prefill" "$ref_decode" "$tol" <<'PY'
import re, sys
run, log = open(sys.argv[1]).read(), open(sys.argv[2]).read()
ref_p, ref_d, tol = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
p = re.search(r"prefill ([0-9.]+) tok/s", run)
d = re.search(r"decode ([0-9.]+) tok/s", run)
calls = [int(c) for c in re.findall(r"R3_MOE_HELPER rank \d+: (\d+) calls", log)]
problems = []
if not p or not d:
    problems.append("no prefill/decode result")
if len(calls) != 2 or min(calls) == 0:
    problems.append(f"helper calls {calls}")
if d and float(d[1]) <= ref_d:
    problems.append(f"decode {d[1]} <= reference {ref_d}")
if p and float(p[1]) < ref_p * (1 - tol):
    problems.append(f"prefill {p[1]} below reference {ref_p} by more than {tol:.0%}")
print("GATE " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
sys.exit(1 if problems else 0)
PY
