#!/bin/bash
# PWR02: XTX helper OverDrive sweep, run from the workstation: max shader clock (knob sclk, MHz) or graphics voltage
# offset (knob vo, mV, negative = undervolt). For each level (in an order that rotates with the default) it sets it on every gfx1100 card of the lab host through sudo (password read once from stdin, passed to
# the remote sudo over ssh stdin, never stored), runs pwr01-baseline.sh there through `bigcherry slice lab`, and restores
# the default clock afterwards (also on exit). Hosts and devices come from the environment settings.
# Usage: <password on stdin> | pwr02-xtx-sclk-sweep.sh <branch with pwr01-baseline.sh> ["levels, 0 = default"] [sclk|vo]
set -u
BRANCH=${1:?branch}; LEVELS=${2:-0 2500 2000 0 2000 2500}; KNOB=${3:-sclk}
IFS= read -r PW
read -r HOST BDFS <<<"$(PYTHONPATH=tools python -c '
from bigcherry.core import environment as e
h = e.load_default().host("brutus-lab")
d = e.load_default().host("build-server").devices_for_arch("gfx1100")
print(h.hostname, " ".join(x.locator for x in d))
')"
[ -n "$HOST" ] && [ -n "$BDFS" ] || { echo "environment: lab host or gfx1100 devices missing"; exit 1; }
set_sclk() {   # $1 MHz, 0 = restore the default
    local cmd=""
    for b in $BDFS; do
        f=/sys/bus/pci/devices/$b/pp_od_clk_voltage
        if [ "$1" = 0 ]; then cmd="$cmd echo r > $f && echo c > $f &&"
        elif [ $KNOB = vo ]; then cmd="$cmd echo 'vo $1' > $f && echo c > $f &&"
        else cmd="$cmd echo 's 1 $1' > $f && echo c > $f &&"; fi
    done
    printf '%s\n' "$PW" | ssh "$HOST" "sudo -S -p '' sh -c \"$cmd true\"" && \
        ssh "$HOST" "for b in $BDFS; do grep -A2 OD_SCLK /sys/bus/pci/devices/\$b/pp_od_clk_voltage | tail -1; grep -A1 OD_VDDGFX_OFFSET /sys/bus/pci/devices/\$b/pp_od_clk_voltage | tail -1; done" </dev/null
}
trap 'set_sclk 0 >/dev/null' EXIT
for lv in $LEVELS; do
    echo "== $KNOB level ${lv} ($( [ $lv = 0 ] && echo default ))"
    set_sclk $lv || { echo "set failed"; exit 1; }
    PYTHONPATH=tools python -m bigcherry slice lab --host brutus-lab "$BRANCH" tools/lab/radiance/pwr01-baseline.sh </dev/null 2>&1 | grep -E "^OUT|PWR01_DONE|error" | tail -2
    echo "LEVEL $lv done"
done
set_sclk 0
echo PWR02_SWEEP_DONE
