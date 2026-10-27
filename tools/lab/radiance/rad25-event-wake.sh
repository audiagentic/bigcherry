#!/bin/bash
# Runner for rad25-event-wake.cpp: yield vs blocking-event completion on one helper-class GPU (the first gfx1100 in the
# host's device inventory, from the environment settings), alternating the two modes over several decode cadences.
# Prints one RESULT line per run and a per-cadence summary (starvation, lane CPU).
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad25-event-wake.sh [repeats=3] [requests=20000]
#   [periods="100 150 300"] [variants=all, or a space-separated subset of the names below]
set -u
REPEATS=${1:-3}; N=${2:-20000}; PERIODS=${3:-100 150 300}; ONLY=${4:-}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
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
# Variants: name|mode|environment. The runtime spins before it blocks a wait; these probe that spin.
VARIANTS=(
  "yield|yield|"
  "block|block|"
  "blockdev|blockdev|"
  "blockonly|blockonly|"
  "block-noactive|block|ROC_ACTIVE_WAIT_TIMEOUT=0"
  "blockdev-noactive|blockdev|ROC_ACTIVE_WAIT_TIMEOUT=0"
  "blockdev-mwaitx|blockdev|HSA_ENABLE_MWAITX=1"
  "blockdev-noactive-mwaitx|blockdev|ROC_ACTIVE_WAIT_TIMEOUT=0 HSA_ENABLE_MWAITX=1"
)
[ -n "$ONLY" ] && { keep=(); for v in "${VARIANTS[@]}"; do [[ " $ONLY " == *" ${v%%|*} "* ]] && keep+=("$v"); done; VARIANTS=("${keep[@]}"); }
for period in $PERIODS; do
    for r in $(seq $REPEATS); do
        n=${#VARIANTS[@]}
        for k in $(seq 0 $((n - 1))); do   # rotate the starting variant each repeat
            IFS='|' read -r name mode envs <<<"${VARIANTS[$(( (k + r) % n ))]}"
            line=$(env $envs ROCR_VISIBLE_DEVICES=$IDX timeout 600 $OUT/rad25-event-wake $mode $N $period 60 4 | grep RESULT)
            echo "${line/RESULT/RESULT variant=$name}" | tee -a $OUT/results.txt
        done
    done
done
python3 - $OUT/results.txt <<'PY' | tee -a $OUT/summary.txt
import re, statistics, sys
rows = [dict(re.findall(r"(\w+)=([\w.-]+)", l)) for l in open(sys.argv[1]) if l.startswith("RESULT")]
variants = list(dict.fromkeys(r["variant"] for r in rows))
for period in sorted({r["period_us"] for r in rows}, key=int):
    for mode in variants:
        g = [r for r in rows if r["period_us"] == period and r["variant"] == mode]
        if not g: continue
        f = lambda k: statistics.median(float(r[k]) for r in g)
        print(f"period {period:>4} us {mode:24s}: starved {f('starved'):7.0f}  starve_total {f('starve_total_us'):9.0f} us  "
              f"p99 {f('starve_p99_us'):7.1f} us  wall/ideal {f('wall_ms')/f('ideal_ms'):.4f}  lane CPU {f('lane_cpu_pct'):5.1f} %")
PY
echo "OUT $OUT"; echo RAD25_WAKE_DONE
