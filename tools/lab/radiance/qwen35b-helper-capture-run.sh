#!/bin/bash
# RCS09/RCS10 genericity check on dev: build the pinned radiance (stock), then serve Qwen3.6-35B-A3B FP8 on the engine alone
# and score output quality (KL) of the XTX workers against the engine alone (qwen35b-helper-capture.json). Run from a checkout of this branch.
set -u
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
export BIGCHERRY_ENVIRONMENT=${BIGCHERRY_ENVIRONMENT:-${BC_REPO_ROOT}/config/environment.local.toml}
WORK=${BC_REPO_ROOT}/work
LOG=$(PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff 2>&1 | grep -v warning | tail -n 4)
echo "$LOG"
PLAN=$(echo "$LOG" | grep -o 'build_plan_id=[0-9a-f]*' | tail -1 | cut -d= -f2)
B=$(ls -d $WORK/builds/*/$PLAN/ 2>/dev/null | head -1)   # the build this run made, not the newest directory
[ -f "${B}bin/radiance" ] || { echo "Q35C_EXIT=99 no build"; exit 99; }
mkdir -p ${BC_CACHE_ROOT}/caps-q35
export KLD_REF_DIR=${KLD_REF_DIR:-$WORK/kld-ref-q35}
export RAD_BIN=${B}bin/radiance RAD_HOME=${B}radiance_home R9700_BDF=0000:09:00.0
OUT=$WORK/q35c-$(date +%m%d-%H%M%S)
echo "OUT=$OUT BIN=$RAD_BIN"
PYTHONPATH=tools timeout 7200 python3 -m bigcherry ab-benchmark --server-config tools/lab/radiance/qwen35b-helper-capture.json --output "$OUT" --settle-seconds 2 2>&1 | grep -v 'warning:'
echo "Q35C_EXIT=${PIPESTATUS[0]}"
