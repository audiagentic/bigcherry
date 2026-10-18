#!/bin/bash
# RAD06: A/B a helper kernel variant on replayed real requests - build the branch, replay the largest captures with
# the variant env off and on, require rel_l2=0 between the helper and the engine reference, and compare the grouped
# GEMM kernel times (rocprofv3 kernel trace, gfx1100).
# Usage: helper-variant-ab.sh <branch> <ENV_NAME> <off-value> <on-value> [captures=4]
set -u
BR=$1; VAR=$2; OFF=$3; ON=$4; N=${5:-4}
W=/mnt/data/bigcherry-work
cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH BIGCHERRY_ENVIRONMENT=/mnt/vault/development/projects/bigcherry/workspaces/main/config/environment.local.toml
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/variant-build.log 2>&1
grep -E 'ok build|FAILED' $W/variant-build.log | tail -1; grep -E ' error:|r3_rewrite.py failed' $W/variant-build.log | sort -u | head
B=$(ls -td /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
M=/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad
for f in $(ls -S $W/captures/*.bin | head -$N); do
  echo "== $(basename $f)"
  for v in $OFF $ON; do
    out=$W/variant-$(basename $f .bin)-$v; rm -rf $out
    env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 R3_MOE_HELPER_MIB=22528 R3_MOE_HELPER_SMALL_ROWS=8 R3_MOE_HELPER_MAX_ROWS=4096 \
      R3_MOE_NATIVE=1 R3_MOE_HELPER_MOE_DOWN_TILE=1 $VAR=$v \
      R3_MOE_HELPER_HEAT=$W/runs/perf51/heat.rank0.hot-first RADIANCE_HOME=$B/radiance_home timeout 900 \
      rocprofv3 --kernel-trace --stats --output-format csv -d $out -o run -- \
      $B/radiance_home/helpers/r3_moe_helper --replay $M $f > $out.log 2>&1
    echo " $VAR=$v: $(grep -h 'engine_join' $out.log | head -1)"
    python3 - $out/run_kernel_stats.csv <<'PY'
import csv, sys
for x in csv.DictReader(open(sys.argv[1])):
    if 'moe_gemm_q_kernel' in x['Name']:
        n = x['Name']; n = n[n.find('<'):n.find('>') + 1]
        print('   %-48s calls %-4s avg %.1f us total %.2f ms' % (n[:48], x['Calls'], float(x['AverageNs']) / 1e3, float(x['TotalDurationNs']) / 1e6))
PY
  done
done
echo VARIANT_DONE
