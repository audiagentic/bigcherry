#!/bin/bash
# Incremental rebuild of the moe worker and the libr3moe plugin inside an existing lane build (seconds, not the 25 minutes a fresh
# lane build takes). The libr3 sub-build compiles straight from the checkout, so: check out the commit, ninja the two targets, copy
# the outputs into the build's radiance_home. Use it for the edit-test loop; the lane build (bigcherry build) stays the formal build.
# Usage: fastbuild.sh <commit> [BUILD_DIR]   (BUILD_DIR defaults to $FASTBUILD_DIR, then the newest build with a libr3 sub-build)
set -eu
SHA=$1
CHECKOUT=${FASTBUILD_CHECKOUT:-/mnt/data/bigcherry-work/wt-accept}
WORK=${WORK:-/mnt/vault/development/projects/bigcherry/workspaces/main/work}
B=${2:-${FASTBUILD_DIR:-$(ls -td $WORK/builds/*/*/ | while read d; do [ -f "${d}components/libr3/build.ninja" ] && { echo "${d%/}"; break; }; done)}}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4} PATH=${ROCM_PATH:-/opt/rocm-7.2.4}/bin:$PATH
git -C "$CHECKOUT" fetch -q origin 2>/dev/null || true
git -C "$CHECKOUT" checkout -q --detach "$SHA"
git -C "$CHECKOUT" log --oneline -1
ninja -C "$B/components/libr3" r3_moe_worker libr3moe 2>&1 | grep -E "error|FAILED|^\[[0-9]+/[0-9]+\] (Linking|Building)" | tail -8
install -m 755 "$B/components/libr3/helpers/r3_moe_worker" "$B/radiance_home/helpers/r3_moe_worker"
install -m 755 "$B/components/libr3/radiance_home/kernels/libr3moe.so" "$B/radiance_home/kernels/libr3moe.so"
echo "FASTBUILD_OK $B"
