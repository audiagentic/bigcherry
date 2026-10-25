#!/bin/bash
# RAD20 decode diagnosis (LAB ONLY): build this slice once, then two 3-arm A/Bs on R3_MOE_WORKER_LANE_DIAG
# (blocking vs poll vs nofence; blocking vs spin vs noflag). Usage: rad20-diag.sh [rounds=6]
set -u
ROUNDS=${1:-6}
W=${BC_WORK_ROOT}
OUT=$W/runs/rad20-diag-$(date +%m%d%H%M); mkdir -p $OUT
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
git rev-parse HEAD | tee $OUT/builds.txt
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
    --experiment moe-remote-handoff > $OUT/build.log 2>&1
plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build.log | tail -1 | cut -d= -f2)
[ -n "$plan" ] || { grep -E ' error:|FAILED' $OUT/build.log | sort -u | head; exit 1; }
B=$(ls -d $W/builds/*/$plan); echo $B | tee -a $OUT/builds.txt
export RAD_BIN_NEW=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0
for n in 1 2; do
    PYTHONPATH=tools timeout 14400 python3 -m bigcherry ab-benchmark --server-config tools/lab/radiance/helper-rad20-diag$n-ab.json \
        --output $OUT/diag$n --pairs $ROUNDS --settle-seconds 20 > $OUT/diag$n.log 2>&1
    tail -1 $OUT/diag$n.log
    for d in $OUT/diag$n/pair-*; do
        python3 - "$d/cell.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); m = c.get("metrics") or {}; dr = c.get("draft") or {}
print(sys.argv[1].split("/")[-2], c.get("position"), " ".join(f"{k[:-4]}={m[k]:.1f}" for k in sorted(m) if k.endswith("_tps") and k != "pp256_tps"),
      dr.get("accepted"), dr.get("drafted"), c.get("metric_error", "")[:80])
PY
    done
done
echo "OUT $OUT"; echo RAD20_DIAG_DONE
