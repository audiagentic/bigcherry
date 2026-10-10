#!/bin/bash
# QFP41 / QFP49: build and run the standalone HIP probes (hip_overlap_probe.cpp) on the cards the queue gives it.
# Run as a queue SCRIPT job on the target cards:
#   VIS=0,1,2 SCRIPT hp1 tools/lab/hip-probes/run.sh @<any build run> <out-dir> [test...]
# The first argument (a llama-server path from the queue) is ignored. Tests: links overlap submit graphs stagger
# (default: all). The cards are the ones in HIP_VISIBLE_DEVICES, which the queue sets from VIS.
# Usage: run.sh <ignored> <out-dir> [test...]
# env: ROCM_PATH (/opt/rocm-7.2.4), PROBE_TIMEOUT (900 s)
set -u
out=$2; shift 2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
if ! hipcc -O2 -std=c++17 -pthread "$here/hip_overlap_probe.cpp" -o "$out/hip_overlap_probe" > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error" "$out/build.log" | head -12 | cut -c1-220; exit 1
fi
echo "cards: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
timeout "${PROBE_TIMEOUT:-900}" "$out/hip_overlap_probe" "$@" 2>&1 | tee "$out/probe.log" | cut -c1-240
echo "probe exit ${PIPESTATUS[0]}"
