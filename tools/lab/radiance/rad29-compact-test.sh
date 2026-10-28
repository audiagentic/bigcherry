#!/bin/bash
# RAD29: build route_compact_test (the wave-aggregated route compaction against the per-route-atomic loop it replaced)
# and run it on one GPU of each architecture in the host's device inventory: equivalence over every case, then --bench
# timing. Usage (from a checkout, e.g. through `bigcherry slice lab`): rad29-compact-test.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad29-compact-$(date +%m%d%H%M); mkdir -p "$OUT"
git log -1 --format='commit %H %s' > "$OUT/summary.txt"
SRC=engines/radiance/components/moe-remote
"$ROCM_PATH/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx1100 --offload-arch=gfx1201 -I"$SRC" \
    "$SRC/route_compact_test.hip" -o "$OUT/route_compact_test" > "$OUT/build.log" 2>&1 || { tail "$OUT/build.log"; exit 1; }
rc=0
for arch in gfx1201 gfx1100; do
    idx=$(PYTHONPATH=tools python3 -c "
from bigcherry.core import environment as e
d = e.load_default().host(None).devices_for_arch('$arch')
print(d[0].index if d else '')")
    [ -n "$idx" ] || { echo "$arch: not in the device inventory" | tee -a "$OUT/summary.txt"; continue; }
    echo "== $arch (device $idx)" | tee -a "$OUT/summary.txt"
    ROCR_VISIBLE_DEVICES=$idx timeout 600 "$OUT/route_compact_test" 2>&1 | tee -a "$OUT/summary.txt" | tail -2 || rc=1
    ROCR_VISIBLE_DEVICES=$idx timeout 900 "$OUT/route_compact_test" --bench > "$OUT/bench-$arch.txt" 2>&1 || rc=1
    tail -1 "$OUT/bench-$arch.txt"
done
echo "OUT $OUT"
exit $rc
