#!/bin/bash
# 1304: one ~80K-deep request with BIGCHERRY_GRAPH_MEMLOG=1; summarise per-device graph instance bytes and counts.
# Usage: graph-memlog.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
BIGCHERRY_GRAPH_MEMLOG=1 DEPTH=65536 DECODE_N=256 bash "$s" "$bin" "$out" timing | grep -E "^timing:"
python3 - "$out/timing.server.log" <<'PY'
import collections, re, sys
per = collections.defaultdict(list)
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"patch=1304_graph_memlog device=(\d+) nodes=(\d+) instance_bytes=(-?\d+) cached=(\d+)", line)
    if m:
        per[int(m.group(1))].append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
for dev, rows in sorted(per.items()):
    b = [r[1] for r in rows]
    print(f"device {dev}: {len(rows)} instantiations, max cached {max(r[2] for r in rows)}, "
          f"instance bytes total {sum(b)/2**20:.1f} MiB, median {sorted(b)[len(b)//2]/2**20:.2f} MiB, "
          f"max {max(b)/2**20:.2f} MiB")
PY
