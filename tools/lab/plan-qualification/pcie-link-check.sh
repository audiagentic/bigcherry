#!/bin/bash
# Check (and optionally retrain) the upstream PCIe links of the AMD GPUs. A root-port link stuck below its
# max speed (seen on Brutus: XTX0 and R9700 at 2.5 GT/s under load, amdgpu pp_dpm_pcie with no active level)
# roughly halves AllReduce-bound prefill until reboot. For each GPU, compares the root port's
# current_link_speed with its max_link_speed (the GPU-internal switch hops do not reflect the slot link).
# Cause seen on Brutus: thermald drives the kernel's PCIe_Port_Link_Speed_<port> cooling devices (bwctrl)
# under CPU heat and never releases them, leaving LnkCtl2 target speed at Gen 1.
# Usage: pcie-link-check.sh            -> report; exit 1 if any root-port link is degraded
#        pcie-link-check.sh --retrain  -> also reset the degraded ports' PCIe_Port_Link_Speed cooling devices
#                                         to 0 and request a link retrain (sudo -S, password on stdin; run
#                                         only with the GPUs idle), wait, and re-check
set -u
retrain=0; [ "${1:-}" = "--retrain" ] && retrain=1
check() {
  bad=()
  for g in $(lspci -D -d 1002: | grep -iE "VGA|Display" | cut -d" " -f1); do
    root=$(readlink -f /sys/bus/pci/devices/$g | tr / "\n" | grep -E "^[0-9a-f]{4}:" | sed -n 1p)
    s=/sys/bus/pci/devices/$root
    cur=$(cat $s/current_link_speed); max=$(cat $s/max_link_speed)
    st=OK; [ "$cur" != "$max" ] && { st=DEGRADED; bad+=("$root"); }
    echo "$g via $root: $cur x$(cat $s/current_link_width) (max $max x$(cat $s/max_link_width)) $st"
  done
}
check
[ ${#bad[@]} -eq 0 ] && exit 0
[ $retrain -eq 1 ] || exit 1
read -r pw
for r in "${bad[@]}"; do
  for c in /sys/class/thermal/cooling_device*; do
    [ "$(cat $c/type)" = "PCIe_Port_Link_Speed_$r" ] || continue
    echo "$pw" | sudo -S -p "" sh -c "echo 0 > $c/cur_state" && echo "cooling device $(basename $c) for $r reset to 0"
  done
  echo "$pw" | sudo -S -p "" setpci -s "$r" CAP_EXP+0x10.w=0x0020:0x0020 && echo "retrain requested on $r"
done
sleep 2
check
[ ${#bad[@]} -eq 0 ]
