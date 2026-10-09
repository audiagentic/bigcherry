#!/bin/bash
# RR01: time radiance's own kernel library (libr4d, gfx12) on the R9700 over the same fixture libr3 is checked with,
# so a libr3 run on gfx1100 can be put beside it kernel by kernel and shape by shape. The two runs are on different
# cards, so the ratio is card plus compatibility layer together, not the layer alone.
# Run as a queue SCRIPT job on the gfx1201 card:
#   VIS=2 SCRIPT r4d-base tools/lab/radiance/r4d-baseline.sh @<any build run> <out-dir> [op...]
# The first argument (a llama-server path from the queue) is ignored. With ops given, only those are run.
# Usage: r4d-baseline.sh <ignored> <out-dir> [op...]
# env: RADIANCE_SRC, GPU (HIP index, 0), COMPARE (a libr3 kbench.json: print its times over this run's)
set -u
out=$2; shift 2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
opts=()
for op in "$@"; do opts+=(--op "$op"); done
echo "radiance $(git -C "$src" rev-parse --short HEAD); libr4d on HIP device ${GPU:-0}"
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$src/build/radiance_home" timeout "${KBENCH_TIMEOUT:-2400}" \
    "$src/build/bin/rad-kbench" --kernels libr4d,libref "${opts[@]}" --bench \
    --fixture "${FIXTURE:-$src/build/radiance_home/kernels.rkb}" --report "$out/kbench.md" > "$out/kbench.log" 2>&1
echo "rad-kbench --kernels libr4d,libref: exit $?"
grep -E "device:|checked,|memory:|refused|not loaded" "$out/kbench.log" | cut -c1-220
[ -f "$out/kbench.md" ] || { echo "NO_REPORT"; tail -8 "$out/kbench.log" | cut -c1-220; exit 1; }
python3 "$here/kdev_report.py" "$out/kbench.md" --json "$out/kbench.json" | grep -A400 "^== per kernel" | cut -c1-230
if [ -n "${COMPARE:-}" ] && [ -f "$COMPARE" ]; then
    echo "== $COMPARE over this run"
    python3 - "$COMPARE" "$out/kbench.json" "$here" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[3])
import kdev_report
kdev_report.compare(json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8")))
PY
fi
exit 0
