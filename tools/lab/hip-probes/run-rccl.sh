#!/bin/bash
# QFP49: build and run the RCCL staggering probe (rccl_stagger_probe.cpp) on the cards the caller makes visible.
# Run as a queue SCRIPT job; SCRIPT jobs do not get HIP_VISIBLE_DEVICES from VIS, so CARDS sets it here:
#   CARDS=0,1,2 ... VIS=0,1,2 SCRIPT rs1 tools/lab/hip-probes/run-rccl.sh @<any build run> <out-dir> [layers passes repeats]
# The first argument (a llama-server path from the queue) is ignored.
# Usage: run-rccl.sh <ignored> <out-dir> [layers [passes [repeats]]]
# env: ROCM_PATH (/opt/rocm-7.2.4), CARDS (0,1,2), PROBE_TIMEOUT (900 s)
set -u
out=$2; shift 2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH" LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${CARDS:-0,1,2}
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
if ! hipcc -O2 -std=c++17 "$here/rccl_stagger_probe.cpp" -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -lrccl -o "$out/rccl_stagger_probe" > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error" "$out/build.log" | head -12 | cut -c1-220; exit 1
fi
echo "cards: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
timeout "${PROBE_TIMEOUT:-900}" "$out/rccl_stagger_probe" "$@" 2>&1 | tee "$out/probe.log" | cut -c1-240
echo "probe exit ${PIPESTATUS[0]}"
