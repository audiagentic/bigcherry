#!/bin/bash
# Runner for rad25-event-wake.cpp: yield vs blocking-event completion on one helper-class GPU (the first gfx1100 in the
# host's device inventory, from the environment settings), alternating the two modes over several decode cadences.
# Prints one RESULT line per run and a per-cadence summary (starvation, lane CPU).
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad25-event-wake.sh [repeats=3] [requests=20000]
set -u
REPEATS=${1:-3}; N=${2:-20000}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad25-event-wake-$(date +%m%d%H%M); mkdir -p $OUT
git log -1 --format='commit %H %s' > $OUT/summary.txt
IDX=$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
d = e.load_default().host(None).devices_for_arch("gfx1100")
print(d[0].index if d else "")
')
[ -n "$IDX" ] || { echo "device inventory: no gfx1100 GPU"; exit 1; }
"$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx1100 tools/lab/radiance/rad25-event-wake.cpp -o $OUT/rad25-event-wake \
    > $OUT/build.log 2>&1 || { tail $OUT/build.log; exit 1; }
for period in 100 150 300; do
    for r in $(seq $REPEATS); do
        order="yield block"; [ $((r % 2)) = 0 ] && order="block yield"   # alternate which mode runs first
        for m in $order; do
            ROCR_VISIBLE_DEVICES=$IDX timeout 600 $OUT/rad25-event-wake $m $N $period 60 4 | tee -a $OUT/results.txt
        done
    done
done
python3 - $OUT/results.txt <<'PY' | tee -a $OUT/summary.txt
import re, statistics, sys
rows = [dict(re.findall(r"(\w+)=([\w.]+)", l)) for l in open(sys.argv[1]) if l.startswith("RESULT")]
for period in sorted({r["period_us"] for r in rows}, key=int):
    for mode in ("yield", "block"):
        g = [r for r in rows if r["period_us"] == period and r["mode"] == mode]
        if not g: continue
        f = lambda k: statistics.median(float(r[k]) for r in g)
        print(f"period {period:>4} us {mode:5s}: starved {f('starved'):7.0f}  starve_total {f('starve_total_us'):9.0f} us  "
              f"p99 {f('starve_p99_us'):7.1f} us  wall/ideal {f('wall_ms')/f('ideal_ms'):.4f}  lane CPU {f('lane_cpu_pct'):5.1f} %")
PY
echo "OUT $OUT"; echo RAD25_WAKE_DONE
