#!/bin/bash
# RAD23: a short Flash-Next probe instead of a long run (~3 minutes). It starts the layout with R3_MOE_WORKER_TRACE=1, sends one
# 42K-token prefill with 64 decode tokens and a mostly cached 14K follow-up, stops, and writes into OUT:
#   NAME.req.log     the two requests' wall time and usage
#   NAME.report.txt  trace_report.py: the engine's exposed wait on the helpers per prefill layer, p50/p95/sum
#   NAME.stages.txt  each helper's large-lane stage times (routes / gate_up / quant / down / reduce) and per-request mean
# Usage: tools/lab/radiance/flashnext-probe.sh OUT NAME BUILD LAYOUT [launcher args, e.g. --budget gfx1030=14336]
# Needs the gpu lease; leaves nothing running.
set -u
OUT=$1; NAME=$2; B=$3; L=$4; shift 4; EXTRA=("$@"); set --
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source tools/env/bigcherry-env.sh >/dev/null 2>&1; [ -n "${BC_WORK_ROOT:-}" ] || { echo "flashnext-probe: BC_WORK_ROOT not set" >&2; exit 1; }
mkdir -p "$OUT"
REQ="$OUT/probe-req.py"
cat > "$REQ" <<'PY'
import glob, json, sys, time, urllib.request
docs = sorted(glob.glob(sys.argv[1] + "/docs/**/*.md", recursive=True))
p, i = "", 0
while len(p) < 120000:
    p += open(docs[i % len(docs)]).read(); i += 1
for n, (chars, gen) in enumerate([(120000, 64), (30000, 1)]):
    b = json.dumps({"prompt": p[:chars], "max_tokens": gen, "temperature": 0}).encode()
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18410/v1/completions", b,
                                                                {"Content-Type": "application/json"}), timeout=3600))
    print(n, round(time.time() - t, 2), d["usage"], flush=True)
PY
R3_MOE_WORKER_TRACE=1 bash tools/lab/radiance/launch-flashnext-best.sh start --build "$B" --layout "$L" --port 18410 "${EXTRA[@]}" > "$OUT/$NAME.launch.log" 2>&1 \
  || { echo "flashnext-probe: launch failed, see $OUT/$NAME.launch.log" >&2; bash tools/lab/radiance/launch-flashnext-best.sh stop > /dev/null 2>&1; exit 1; }
python3 "$REQ" "$PWD" > "$OUT/$NAME.req.log" 2>&1
bash tools/lab/radiance/launch-flashnext-best.sh stop > "$OUT/$NAME.stop.log" 2>&1
G=$(grep -o "$BC_WORK_ROOT/flashnext/helper\*-[0-9-]*\.log" "$OUT/$NAME.launch.log" | head -1)
python3 tools/lab/radiance/heat-placement/trace_report.py $(ls $G) > "$OUT/$NAME.report.txt" 2>&1
for f in $(ls $G); do grep -E "stages \(us/request\)|served" "$f" | sed "s#^#$(basename "$f" | cut -d- -f1) #"; done > "$OUT/$NAME.stages.txt"
cat "$OUT/$NAME.req.log"; sed -n 2,7p "$OUT/$NAME.report.txt"; cat "$OUT/$NAME.stages.txt"
