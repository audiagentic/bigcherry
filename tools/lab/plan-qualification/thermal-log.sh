#!/bin/bash
# Log CPU package / hottest-core temperature (coretemp) and the GPU root-port PCIe link speeds every 5 s
# until <stop-file-pattern> appears in <log>, e.g. to watch a CPU-heavy run with thermald disabled.
# Usage: thermal-log.sh <out.csv> <log-to-watch> <stop-pattern>
set -u
out=$1 watch=$2 pat=$3
hw=$(grep -l coretemp /sys/class/hwmon/*/name | head -1 | xargs dirname)
crit=$(( $(cat $hw/temp1_crit 2>/dev/null || echo 100000) / 1000 ))
echo "time,pkg_c,max_core_c,tjmax_c,xtx0,xtx1,r9700" > "$out"
until grep -q "$pat" "$watch" 2>/dev/null; do
  pkg=$(( $(cat $hw/temp1_input) / 1000 ))
  mx=0; for t in $hw/temp*_input; do v=$(( $(cat $t) / 1000 )); [ $v -gt $mx ] && mx=$v; done
  l() { cat /sys/bus/pci/devices/0000:$1/current_link_speed | cut -d" " -f1; }
  echo "$(date +%T),$pkg,$mx,$crit,$(l 00:01.0),$(l 00:01.1),$(l 00:06.0)" >> "$out"
  sleep 5
done
awk -F, 'NR>1{if($2>p)p=$2;if($3>m)m=$3; if($5!="16.0"||$6!="16.0"||$7!="16.0")d++} END{print "THERMAL_SUMMARY max_pkg="p"C max_core="m"C samples="NR-1" degraded_link_samples="d+0}' "$out" | tee -a "$out"
