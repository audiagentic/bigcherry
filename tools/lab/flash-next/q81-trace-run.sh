#!/bin/bash
# One short traced decode (BIGCHERRY_Q81_TRACE=1) and a summary of Q8_1 cache publishes vs MMVQ misses.
# Usage: q81-trace-run.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
here=$(cd "$(dirname "$0")" && pwd)
BIGCHERRY_Q81_TRACE=1 DEPTH=8192 DECODE_N=32 bash "$here/long-ctx-profile.sh" "$bin" "$out" timing 2>&1 | grep -E "^timing"
log=$out/timing.server.log
echo "publish-rms: $(grep -c 'BIGCHERRY_Q81 publish-rms' $log)  publish-act: $(grep -c 'BIGCHERRY_Q81 publish-act' $log)  misses: $(grep -c 'BIGCHERRY_Q81 miss' $log)"
echo "== sample publish-rms"; grep -m 4 'BIGCHERRY_Q81 publish-rms' $log | cut -c1-220
echo "== misses whose data matches a publish (same buffer, different key?)"
python3 - "$log" <<'PY'
import re, sys, collections
pubs = {}
misses = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"publish-(rms|act) gen=(\d+) node=(\S+?)\((.*?)\) data=(\S+) ne=(\S+)", line)
    if m:
        pubs[(m.group(2), m.group(5))] = (m.group(1), m.group(3), m.group(4), m.group(6))
        continue
    m = re.search(r"miss gen=(\d+) src1=(\S+?)\((.*?) op=(\S+) view_of=(\S+)\) data=(\S+) ne=(\S+)", line)
    if m:
        misses.append(m.groups())
same_buf = collections.Counter()
for gen, node, name, op, view_of, data, ne in misses:
    p = pubs.get((gen, data))
    if p:
        same_buf[(p[0], p[2], name, op, view_of, p[3], ne)] += 1
for k, c in same_buf.most_common(8):
    print(c, "publish", k[0], k[1], "ne", k[5], "<-> miss", k[2], "op", k[3], "view_of", k[4], "ne", k[6])
gens_pub = {g for (g, _) in pubs}
print("misses in a generation with any publish:", sum(1 for m in misses if m[0] in gens_pub), "of", len(misses))
print("miss ops:", collections.Counter(m[3] for m in misses).most_common(6))
PY
