#!/bin/bash
# RAD06: long-run tile split (R3_MOE_WORKER_MOE_LONG_ROWS) - build, replay captured requests with the split off and on,
# require rel_l2=0 between the two, and compare the grouped GEMM kernel times (gfx1100, rocprofv3 kernel trace).
# Usage: helper-long-rows.sh <branch> [long_rows=65]
set -u
BR=${1:-origin/feat/rad06-gemm-counters}; LR=${2:-65}
W=${BC_WORK_ROOT}
cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
export BIGCHERRY_ENVIRONMENT=${BC_DEV_ROOT}/projects/bigcherry/workspaces/main/config/environment.local.toml; source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"; export PATH=$ROCM_PATH/bin:$PATH
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/longrows-build.log 2>&1
grep -E 'ok build|FAILED' $W/longrows-build.log | tail -1; grep -E ' error:|spill|rewrite' $W/longrows-build.log | sort -u | head
B=$(ls -td ${BC_DEV_ROOT}/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
M=${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad
for f in $(ls -S $W/captures/*.bin | head -4); do
  echo "== $(basename $f) $(stat -c %s $f) bytes"
  for lr in 0 $LR; do
    out=$W/longrows-$(basename $f .bin)-$lr; rm -rf $out
    HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 R3_MOE_WORKER_MIB=22528 R3_MOE_WORKER_SMALL_ROWS=8 R3_MOE_WORKER_MAX_ROWS=4096 \
      R3_MOE_NATIVE=1 R3_MOE_WORKER_MOE_DOWN_TILE=1 R3_MOE_WORKER_MOE_LONG_ROWS=$lr \
      R3_MOE_WORKER_HEAT=$W/runs/perf51/heat.rank0.hot-first RADIANCE_HOME=$B/radiance_home timeout 900 \
      rocprofv3 --kernel-trace --stats --output-format csv -d $out -o run -- \
      $B/radiance_home/helpers/r3_moe_worker --replay $M $f > $out.log 2>&1
    echo " long_rows=$lr: $(grep -h 'replay layer\|engine_join' $out.log | head -2 | tr '\n' ' ')"
    python3 - $out/run_kernel_stats.csv <<'PY'
import csv, sys
for x in csv.DictReader(open(sys.argv[1])):
    if 'moe_gemm_q_kernel' in x['Name'] or 'moe_blockmap' in x['Name']:
        n = x['Name']; n = n[n.find('<'):n.find('>') + 1] if '<' in n else n
        print('   %-62s calls %-4s avg %.1f us total %.2f ms' % (n[:62], x['Calls'], float(x['AverageNs']) / 1e3, float(x['TotalDurationNs']) / 1e6))
PY
  done
done
echo LONGROWS_DONE
