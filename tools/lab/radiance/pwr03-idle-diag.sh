#!/bin/bash
# PWR03 diagnosis: with the helpers on the efficiency cores, CPU package power stays at 10-40 W after requests end
# (9 W with default placement). Starts the layout with the helpers pinned (cores found as in pwr03-ecore-ab.sh), serves a
# few requests, then samples every thread's CPU use (top -H) and package power for 60 s of idle.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): pwr03-idle-diag.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/pwr03-idle-diag-$(date +%m%d%H%M); mkdir -p $OUT
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
[ -n "$ECORES" ] || { echo "no efficiency cores found: skipped"; exit 0; }
source tools/lab/radiance/lab-build.sh
B=$(rad_build $OUT/build.log) || exit 1
L=tools/lab/radiance/launch-flashnext-best.sh
URL=http://127.0.0.1:36080/v1/completions   # the launcher's default port
HELPER_WRAP="taskset -c $ECORES" bash $L start --build $B > $OUT/launch.log 2>&1 || { tail $OUT/launch.log; bash $L stop; exit 1; }
for i in 1 2 3 4; do curl -s -m 120 $URL -d "{\"prompt\":\"Story $i about a ship.\",\"max_tokens\":128,\"temperature\":0}" > /dev/null & done; wait
sleep 5
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
e0=$(cat $RAPL); top -H -b -d 5 -n 12 -w 200 > $OUT/top.txt; e1=$(cat $RAPL)
echo "idle package power over 60 s: $(python3 -c "import sys; print(round((int(sys.argv[2])-int(sys.argv[1]))/1e6/60,1))" $e0 $e1) W" | tee $OUT/summary.txt
# The threads that used CPU while idle, summed over the samples.
awk '/^ *PID/{s=1; next} s && NF>11 && $9+0>0.5 {k=$12" "$1; c[k]+=$9} END{for (k in c) printf "%6.1f %s\n", c[k]/12, k}' $OUT/top.txt | sort -rn | head -15 | tee -a $OUT/summary.txt
bash $L stop > $OUT/stop.log 2>&1
echo "OUT $OUT"; echo PWR03_IDLE_DIAG_DONE
