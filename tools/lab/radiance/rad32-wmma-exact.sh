#!/bin/bash
# RAD32: builds and runs rad32-wmma-exact.hip on the first gfx1100 card: does the gfx11 bf16 WMMA sum small exact
# E4M3 x E2M1 products exactly? Usage (through `bigcherry slice lab`): rad32-wmma-exact.sh [trials=2000]
set -u
TRIALS=${1:-2000}
set --
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad32-wmma-exact-$(date +%m%d%H%M); mkdir -p "$OUT"
"$ROCM_PATH/bin/hipcc" -O2 --offload-arch=gfx1100 tools/lab/radiance/rad32-wmma-exact.hip -o "$OUT/probe" > "$OUT/build.log" 2>&1 \
    || { tail -20 "$OUT/build.log"; exit 1; }
IDX=$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
d = e.load_default().host(None).devices_for_arch("gfx1100")
print(d[0].index if d else "")')
ROCR_VISIBLE_DEVICES=$IDX "$OUT/probe" "$TRIALS" | tee "$OUT/result.txt"
