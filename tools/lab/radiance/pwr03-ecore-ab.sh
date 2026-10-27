#!/bin/bash
# PWR03: does running the helper processes on the CPU's efficiency cores cut package power without slowing the model?
# Runs pwr01-baseline.sh in the order default, E-cores, E-cores, default; the E-core arm starts each helper under
# `taskset -c <E-cores>` through the launcher's HELPER_WRAP. The E-core list is read from the kernel (cpu_atom), so a
# CPU without E-cores stops here. Compare the arms' cpu_pkg watts and tok/s in the printed summaries.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): pwr03-ecore-ab.sh
set -u
ECORES=$(cat /sys/devices/cpu_atom/cpus 2>/dev/null)
[ -n "$ECORES" ] || { echo "no efficiency cores on this CPU (no /sys/devices/cpu_atom/cpus)"; exit 1; }
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
