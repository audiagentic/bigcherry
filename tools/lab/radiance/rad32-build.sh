#!/bin/bash
# RAD32: build this checkout's radiance pin with the helper hand-off (the moe-remote-handoff experiment: patches applied,
# libr3moe and the helper compiled against that pin's ABI) on the lab host, reusing a build of the same source tree.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad32-build.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-build.sh"
OUT=$BC_WORK_ROOT/runs/rad32-build-$(date +%m%d%H%M); mkdir -p "$OUT"
git log -1 --format='commit %H %s' | tee "$OUT/summary.txt"
grep -E '^pinned' engines/radiance/recipes.toml | tee -a "$OUT/summary.txt"
if B=$(rad_build "$OUT/build.log"); then
    echo "BUILD_OK $B" | tee -a "$OUT/summary.txt"
    ls "$B/bin/radiance" "$B/radiance_home/helpers/r3_moe_worker" | tee -a "$OUT/summary.txt"
else
    echo "BUILD_FAILED (log $OUT/build.log)" | tee -a "$OUT/summary.txt"
    grep -E ' error:|FAILED|error ' "$OUT/build.log" | sort -u | head -20
    exit 1
fi
