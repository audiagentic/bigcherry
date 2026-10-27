#!/bin/bash
# PWR02: XTX helper OverDrive sweep, run from the workstation: max shader clock (knob sclk, MHz) or graphics voltage
# offset (knob vo, mV, negative = undervolt). For each level (in an order that rotates with the default) it sets it on every gfx1100 card of the lab host through sudo (password read once from stdin, passed to
# the remote sudo over ssh stdin, never stored), runs pwr01-baseline.sh there through `bigcherry slice lab`, and restores
# the default clock afterwards (also on exit). Hosts and devices come from the environment settings.
# Knob mclk sets the OD memory clock max (MHz); PWR02_CAP=<W> holds a power cap through an mclk sweep.
# PWR02_VO=<mV> holds a voltage offset through a clock sweep (a clock write resets it).
# PWR02_RUN_SCRIPT picks the lab script run at each level (default pwr01-baseline.sh; pwr01-soak.sh for a soak).
# Usage: <password on stdin> | BIGCHERRY_LAB_HOST=<lab host entry> pwr02-xtx-sclk-sweep.sh <branch with pwr01-baseline.sh> ["levels, 0 = default"] [sclk|vo]
set -u
BRANCH=${1:?branch}; LEVELS=${2:-0 2500 2000 0 2000 2500}; KNOB=${3:-sclk}
# PWR02_ARCH picks the cards by architecture (default the gfx1100 helpers; gfx1201 for the engine card). Knob cap sets
# the board power cap in watts (restored from power1_cap_default).
ARCH=${PWR02_ARCH:-gfx1100}
IFS= read -r PW
# The lab host is the configured [host.*] entry named by BIGCHERRY_LAB_HOST (as for `bigcherry slice lab --host`); its
# gfx1100 devices come from whichever configured host entry for the same machine carries the device inventory.
LAB=${BIGCHERRY_LAB_HOST:?set BIGCHERRY_LAB_HOST to the configured lab host entry}
read -r HOST BDFS <<<"$(PYTHONPATH=tools python -c '
import sys
from bigcherry.core import environment as e
env = e.load_default()
lab = env.host(sys.argv[1])
devs = [d for h in env.hosts.values() if h.hostname == lab.hostname for d in h.devices_for_arch(sys.argv[2])]
print(lab.hostname, " ".join(sorted({d.locator for d in devs})))
' "$LAB" "$ARCH")"
[ -n "$HOST" ] && [ -n "$BDFS" ] || { echo "environment: lab host or $ARCH devices missing"; exit 1; }
set_sclk() {   # $1 MHz, 0 = restore the default
    local cmd=""
    for b in $BDFS; do
        f=/sys/bus/pci/devices/$b/pp_od_clk_voltage
        # The card's hwmon directory, resolved first: sh does not expand a wildcard in a redirection target.
        h=$(ssh "$HOST" "ls -d /sys/bus/pci/devices/$b/hwmon/hwmon* | head -1" </dev/null)
        if [ "$1" = 0 ]; then cmd="$cmd echo r > $f && echo c > $f && cat $h/power1_cap_default > $h/power1_cap &&"
        elif [ $KNOB = vo ]; then cmd="$cmd echo 'vo $1' > $f && echo c > $f &&"
        elif [ $KNOB = cap ]; then cmd="$cmd echo $(( $1 * 1000000 )) > $h/power1_cap &&"
        # Memory clock (knob mclk, MHz): OD_MCLK max; holds PWR02_VO like the clock knob, and PWR02_CAP (W) when set.
        elif [ $KNOB = mclk ]; then cmd="$cmd echo 'm 1 $1' > $f && ${PWR02_VO:+echo 'vo $PWR02_VO' > $f &&} echo c > $f && ${PWR02_CAP:+echo $(( ${PWR02_CAP:-0} * 1000000 )) > $h/power1_cap &&}"
        # A clock write and commit resets the voltage offset on these cards, so PWR02_VO (mV) is staged with every clock
        # level when a sweep must hold an undervolt.
        else cmd="$cmd echo 's 1 $1' > $f && ${PWR02_VO:+echo 'vo $PWR02_VO' > $f &&} echo c > $f &&"; fi
    done
    printf '%s\n' "$PW" | ssh "$HOST" "sudo -S -p '' sh -c \"$cmd true\"" && \
        ssh "$HOST" "for b in $BDFS; do grep -A2 OD_SCLK /sys/bus/pci/devices/\$b/pp_od_clk_voltage | tail -1; grep -A1 OD_VDDGFX_OFFSET /sys/bus/pci/devices/\$b/pp_od_clk_voltage | tail -1; echo mclk-levels \$(tail -1 /sys/bus/pci/devices/\$b/pp_dpm_mclk); echo cap \$(cat /sys/bus/pci/devices/\$b/hwmon/hwmon*/power1_cap); done" </dev/null
}
trap 'set_sclk 0 >/dev/null' EXIT
for lv in $LEVELS; do
    echo "== $KNOB level ${lv} ($( [ $lv = 0 ] && echo default ))"
    set_sclk $lv || { echo "set failed"; exit 1; }
    PYTHONPATH=tools python -m bigcherry slice lab --host "$LAB" "$BRANCH" "${PWR02_RUN_SCRIPT:-tools/lab/radiance/pwr01-baseline.sh}" </dev/null 2>&1 | grep -E "^OUT|PWR01_DONE|error" | tail -2
    echo "LEVEL $lv done"
done
set_sclk 0
echo PWR02_SWEEP_DONE
