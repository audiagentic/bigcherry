#!/bin/bash
# RCS09/RCS10 genericity check on dev: build the pinned radiance (stock), then serve Qwen3.6-35B-A3B FP8 on the engine alone
# and run the 1/2/4/6-stream baseline (qwen35b-engine-ab.json). Run from a checkout of this branch.
set -u
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
export BIGCHERRY_ENVIRONMENT=${BIGCHERRY_ENVIRONMENT:-${BC_REPO_ROOT}/config/environment.local.toml}
WORK=${BC_REPO_ROOT}/work
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance 2>&1 | grep -v warning | tail -n 4
B=$(ls -td $WORK/builds/*/*/ 2>/dev/null | head -1)
[ -f "${B}bin/radiance" ] || { echo "Q35_EXIT=99 no build"; exit 99; }
export RAD_BIN=${B}bin/radiance RAD_HOME=${B}radiance_home R9700_BDF=0000:09:00.0
OUT=$WORK/q35-$(date +%m%d-%H%M%S)
echo "OUT=$OUT BIN=$RAD_BIN"
PYTHONPATH=tools timeout 7200 python3 -m bigcherry ab-benchmark --server-config tools/lab/radiance/qwen35b-engine-ab.json --output "$OUT" --pairs 2 --settle-seconds 2 2>&1 | grep -v 'warning:'
echo "Q35_EXIT=${PIPESTATUS[0]}"
