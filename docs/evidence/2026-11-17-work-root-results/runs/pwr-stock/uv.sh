#!/bin/bash
D=<work-root>/runs/pwr-stock; X=<work-root>/runs/xtx-undervolt-ab
trap 'python3 $X/txn.py restore uv125-pwr01 >> $D/uv.txn.log 2>&1' EXIT
python3 $X/txn.py apply uv125-pwr01 > $D/uv.txn.log 2>&1 || { echo "apply failed" > $D/uv.status; exit 1; }
for c in /sys/class/drm/card0/device /sys/class/drm/card1/device; do grep -A1 OD_VDDGFX_OFFSET $c/pp_od_clk_voltage | tail -n1; done >> $D/uv.txn.log
bash $D/run.sh
cp $D/out.log $D/uv.out.log
echo done > $D/uv.status
