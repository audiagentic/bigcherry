#!/bin/bash
# RAD06: end-to-end A/B of R3_MOE_FP16 off vs on (helper-fp16-ab.json): build the branch, then ab-benchmark.
# Usage: helper-fp16-e2e.sh <branch> [pairs=2]
set -u
BR=${1:-origin/feat/rad06-fp16-wmma}; PAIRS=${2:-2}
W=${BC_WORK_ROOT}
cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH BIGCHERRY_ENVIRONMENT=${BC_DEV_ROOT}/projects/bigcherry/workspaces/main/config/environment.local.toml
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/fp16e2e-build.log 2>&1
grep -E 'ok build|FAILED' $W/fp16e2e-build.log | tail -1; grep -E ' error:|r3_rewrite.py failed' $W/fp16e2e-build.log | sort -u | head
B=$(ls -td ${BC_DEV_ROOT}/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0
OUT=$W/runs/rad06-fp16-$(date +%m%d%H%M)
PYTHONPATH=tools timeout 9000 python3 -m bigcherry ab-benchmark --server-config tools/lab/radiance/helper-fp16-ab.json \
  --output $OUT --pairs $PAIRS --settle-seconds 20 2>&1 | tail -2
for d in $OUT/pair-*; do echo "$(basename $d) $(grep -oE '"(pp1024|pp4096|tg512|tg2048)_tps": *[0-9.]+' $d/cell.json | tr '\n' ' ')"; done
echo FP16E2E_DONE
