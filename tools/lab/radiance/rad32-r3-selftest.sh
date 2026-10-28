#!/bin/bash
# RAD32: libr3's kernel selftest on one gfx1100 card of the device inventory, against the build of this checkout's
# radiance pin (rad_build reuses it): builds the r3_selftest target in that build's libr3 tree and runs the catalogue
# (every row against its host reference). Usage (from a checkout, e.g. through `bigcherry slice lab`): rad32-r3-selftest.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-build.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad32-r3-selftest-$(date +%m%d%H%M); mkdir -p "$OUT"
git log -1 --format='commit %H %s' | tee "$OUT/summary.txt"
B=$(rad_build "$OUT/build.log") || { echo "BUILD_FAILED"; exit 1; }
cmake --build "$B/components/libr3" --target r3_selftest > "$OUT/selftest-build.log" 2>&1 \
    || { grep -E 'error' "$OUT/selftest-build.log" | sort -u | head -20; exit 1; }
SELF=$(find "$B/components/libr3" -maxdepth 2 -name r3_selftest -type f | head -1)
idx=$(PYTHONPATH=tools python3 -c "
from bigcherry.core import environment as e
d = e.load_default().host(None).devices_for_arch('gfx1100')
print(d[0].index if d else '')")
[ -n "$idx" ] || { echo "no gfx1100 card in the device inventory"; exit 1; }
echo "build $B, selftest $SELF, device $idx" | tee -a "$OUT/summary.txt"
ROCR_VISIBLE_DEVICES=$idx timeout 3600 "$SELF" "$B/radiance_home/kernels/libr3.so" > "$OUT/selftest.txt" 2>&1
rc=$?
grep -iE "fail|mismatch|error" "$OUT/selftest.txt" | head -30 | tee -a "$OUT/summary.txt"
tail -5 "$OUT/selftest.txt" | tee -a "$OUT/summary.txt"
echo "OUT $OUT rc=$rc"
exit $rc
