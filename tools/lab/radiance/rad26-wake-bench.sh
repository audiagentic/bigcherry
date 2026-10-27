#!/bin/bash
# Runner for rad26-wake-bench.cpp on the engine-class GPU (the gfx1201 card in the host's device inventory): spin vs
# hostfunc wake round trips at several request spacings, and the engine-stream cost of a host callback.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad26-wake-bench.sh [repeats=3]
set -u
REPEATS=${1:-3}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad26-wake-bench-$(date +%m%d%H%M); mkdir -p $OUT
git log -1 --format='commit %H %s' > $OUT/summary.txt
IDX=$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
d = e.load_default().host(None).devices_for_arch("gfx1201")
print(d[0].index if d else "")
')
[ -n "$IDX" ] || { echo "device inventory: no gfx1201 GPU"; exit 1; }
"$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx1201 tools/lab/radiance/rad26-wake-bench.cpp -o $OUT/rad26-wake-bench \
    > $OUT/build.log 2>&1 || { tail $OUT/build.log; exit 1; }
for r in $(seq $REPEATS); do
    for gap in 300 1000 3000; do
        for m in spin hostfunc sidefunc; do
            ROCR_VISIBLE_DEVICES=$IDX timeout 300 $OUT/rad26-wake-bench $m 2000 $gap 50 | tee -a $OUT/results.txt
        done
    done
    ROCR_VISIBLE_DEVICES=$IDX timeout 300 $OUT/rad26-wake-bench stall 5000 0 50 | tee -a $OUT/results.txt
done
python3 - $OUT/results.txt <<'PY' | tee -a $OUT/summary.txt
import re, statistics, sys
rows = [dict(re.findall(r"(\w+)=([\w.-]+)", l)) for l in open(sys.argv[1]) if l.startswith("RESULT")]
med = lambda g, k: statistics.median(float(r[k]) for r in g)
for gap in sorted({r["gap_us"] for r in rows if r["mode"] != "stall"}, key=int):
    for m in ("spin", "hostfunc", "sidefunc"):
        g = [r for r in rows if r["mode"] == m and r["gap_us"] == gap]
        if g: print(f"gap {gap:>5} us {m:8s}: round trip p50 {med(g,'rt_p50_us'):7.1f} p90 {med(g,'rt_p90_us'):7.1f} "
                    f"p99 {med(g,'rt_p99_us'):7.1f} us  helper CPU {med(g,'helper_cpu_pct'):5.1f} %")
g = [r for r in rows if r["mode"] == "stall"]
if g: print(f"engine stream: {med(g,'stall_us_per_callback'):.2f} us per host callback, {med(g,'side_stall_us_per_callback'):.2f} us on a side stream "
            f"(work {med(g,'base_us_per_iter'):.1f} us/iter alone, {med(g,'hostfunc_us_per_iter'):.1f} with a callback)")
PY
echo "OUT $OUT"; echo RAD26_WAKE_DONE
