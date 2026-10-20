#!/bin/bash
# RAD15: engine-alone 35B A/B of bc-gfx12's moe_gemm_q decode row (qwen35b-bcgfx12-ab.json). BUILD_DIR: a stock radiance build
# (its bin/ and radiance_home/); AB_CONFIG=tools/lab/radiance/qwen35b-bcgfx12-kld.json for the KL check; BCG12_SO: the built libbcgfx12.so. The plugin is added to a copy of the build's radiance home.
set -u
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
export BIGCHERRY_ENVIRONMENT=${BIGCHERRY_ENVIRONMENT:-/mnt/vault/development/projects/bigcherry/workspaces/main/config/environment.local.toml}
WORK=/mnt/vault/development/projects/bigcherry/workspaces/main/work
B=${BUILD_DIR%/}
[ -x "$B/bin/radiance" ] && [ -f "$BCG12_SO" ] || { echo "BCG12_EXIT=99 need BUILD_DIR and BCG12_SO"; exit 99; }
H=$WORK/bcg12-home-$(date +%m%d-%H%M%S)
cp -r "$B/radiance_home" "$H" && cp "$BCG12_SO" "$H/kernels/"
export KLD_REF_DIR=${KLD_REF_DIR:-$WORK/kld-ref-q35}
export RAD_BIN=$B/bin/radiance RAD_HOME=$H R9700_BDF=0000:09:00.0
OUT=$WORK/q35-bcg12-$(date +%m%d-%H%M%S)
echo "OUT=$OUT BIN=$RAD_BIN HOME=$H"
PYTHONPATH=tools timeout 7200 python3 -m bigcherry ab-benchmark --server-config ${AB_CONFIG:-tools/lab/radiance/qwen35b-bcgfx12-ab.json} \
  --output "$OUT" --pairs "${PAIRS:-2}" --settle-seconds 2 2>&1 | grep -v 'warning:'
echo "BCG12_EXIT=${PIPESTATUS[0]}"
