#!/bin/bash
# PWR03: does running the helper processes on the CPU's efficiency cores cut package power without slowing the model?
# Runs pwr01-baseline.sh in the order default, E-cores, E-cores, default; the E-core arm starts each helper under
# `taskset -c <E-cores>` through the launcher's HELPER_WRAP. The E-core list is read from the kernel (generic, below),
# so a CPU without efficiency cores is skipped. Compare the arms' cpu_pkg watts and tok/s in the printed summaries.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): pwr03-ecore-ab.sh
set -u
# Efficiency cores, found generically: CPUs whose kernel cpu_capacity is below the largest (any asymmetric CPU, e.g.
# ARM big.LITTLE or AMD/Intel parts that report it), else the Intel hybrid "atom" PMU's CPU list. None found: nothing to
# compare on this host.
ECORES=$(python3 - <<'PY'
import glob, os
cap = {}
for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpu_capacity"):
    try: cap[int(p.split("/")[-2][3:])] = int(open(p).read())
    except (OSError, ValueError): pass
if cap and min(cap.values()) < max(cap.values()):
    print(",".join(str(c) for c in sorted(cap) if cap[c] < max(cap.values())))
elif os.path.exists("/sys/devices/cpu_atom/cpus"):
    print(open("/sys/devices/cpu_atom/cpus").read().strip())
PY
)
[ -n "$ECORES" ] || { echo "no efficiency cores found on this CPU (uniform cpu_capacity, no hybrid PMU): skipped"; exit 0; }
echo "E-cores: $ECORES"
for arm in default ecore ecore default; do
    echo "== arm $arm"
    if [ $arm = ecore ]; then
        HELPER_WRAP="taskset -c $ECORES" bash tools/lab/radiance/pwr01-baseline.sh | grep -E "^cpu_pkg|^gpu|tok/s|^OUT|decode text"
    else
        bash tools/lab/radiance/pwr01-baseline.sh | grep -E "^cpu_pkg|^gpu|tok/s|^OUT|decode text"
    fi
done
echo PWR03_ECORE_DONE
