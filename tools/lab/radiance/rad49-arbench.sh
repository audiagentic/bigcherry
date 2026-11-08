#!/bin/bash
# RAD49 step 1: the two-rank exchange alone on the two RX 7900 XTX (r3-arbench.cpp: libr3's pinned-host-memory ar_hc_write and
# all_reduce rows through the plugin ABI, no model), at M = 1, 4, 24, 32 and 2048 rows of 2560, exact and wht6, first with the
# R9700 idle and then while it pulls from host memory at full rate (host-pull.hip), as a remote-MoE helper does.
# The end-to-end variant (the same kernels inside the engine, --profile-ops) is the task's arbench.sh, kept in its scratch folder.
#
# Usage: B=<radiance lane build dir> bash tools/lab/radiance/rad49-arbench.sh
#   Env: TTL (lease, default 2h), ROWS (default 1,4,24,32,2048), ITERS (default 200), BUILD_DIR (harness builds,
#        default $BC_WORK_ROOT/builds-rad49/arbench).
# Output: $BC_WORK_ROOT/runs/rad49-arbench/harness-<stamp>/ : pull0.tsv, pull1.tsv, pull1-host-pull.log, summary.txt.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
TTL=${TTL:-2h}

# The gpu lease for the whole run (it refuses while any other GPU process is running).
if [ -z "${R3_ARBENCH_IN_LEASE:-}" ]; then
  exec agent-lease acquire gpu --ttl "$TTL" --purpose "RAD49 r3-arbench: two-rank XTX exchange timing (exact/wht6, R9700 pull on/off)" \
    -- env R3_ARBENCH_IN_LEASE=1 bash "$HERE/$(basename "${BASH_SOURCE[0]}")" "$@"
fi

cd "$REPO" || exit 1
source tools/env/bigcherry-env.sh >/dev/null
B=${B:?set B to a radiance lane build directory}
LIB=$B/radiance_home/kernels/libr3.so
INC=$B/runtime-prefix/include/radiance
# The ROCm SDK the build was configured with (its recorded HIP compiler), never the unversioned system one.
ROCM=$(sed -n 's#^CMAKE_HIP_COMPILER:[A-Z]*=\(.*\)/lib/llvm/bin/clang++$#\1#p' "$B/CMakeCache.txt")
[ -d "$ROCM/include/hip" ] || { echo "no recorded ROCm SDK in $B/CMakeCache.txt" >&2; exit 1; }
[ -f "$LIB" ] && [ -f "$INC/rad_abi.h" ] || { echo "no libr3.so / ABI headers in $B" >&2; exit 1; }
# The message shape is the served model's: n_embd and hc_count from its container (the layout's model), never a literal here.
MODEL=${MODEL:-$BC_MODEL_ROOT/$(python3 -c "import tomllib,sys;print(tomllib.load(open(sys.argv[1],'rb'))['layout']['flashnext-tp2xtx-r9700']['model'])" engines/radiance/layouts.toml)}
read -r NEMBD HCN < <("$B/bin/rad-info" "$MODEL" | awk '$1=="n_embd"{n=$2} $1=="hc_count"{h=$2} END{print n, h}')
[ -n "$NEMBD" ] && [ -n "$HCN" ] || { echo "no n_embd / hc_count in $MODEL" >&2; exit 1; }
ROWS=${ROWS:-1,4,24,32,2048}
ITERS=${ITERS:-200}
OUT=$BC_WORK_ROOT/runs/rad49-arbench/harness-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ git -C "$REPO" log -1 --format='checkout %H %s'; echo "build $B"; echo "model $MODEL n_embd=$NEMBD hc_count=$HCN"; } > "$OUT/summary.txt"

# The harness is built against the ABI headers of the build it loads (it refuses a plugin of another ABI version): one build
# per header digest, card-free, reused across runs.
digest=$(cat "$INC"/rad_abi.h "$INC"/rad_types.h "$HERE/r3-arbench.cpp" | sha256sum | cut -c1-12)
BIN=${BUILD_DIR:-$BC_WORK_ROOT/builds-rad49/arbench}/$digest
if [ ! -x "$BIN/r3-arbench" ] || [ ! -x "$BIN/host-pull" ]; then
  mkdir -p "$BIN"
  CXX=${CXX:-$(command -v g++-14 || command -v g++)}
  "$CXX" -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I"$INC" -I"$ROCM/include" "$HERE/r3-arbench.cpp" \
    -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lamdhip64 -ldl -lpthread -o "$BIN/r3-arbench" || { echo "r3-arbench build failed" >&2; exit 1; }
  "$ROCM/bin/hipcc" -O3 --offload-arch=gfx1201 "$HERE/host-pull.hip" -o "$BIN/host-pull" || { echo "host-pull build failed" >&2; exit 1; }
fi

# Which cards: the host inventory, never a literal index.
read -r XTX R9700 < <(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
h = e.load_default().host(None)
x = [d.index for d in h.devices_for_arch("gfx1100")]
r = [d.index for d in h.devices_for_arch("gfx1201")]
assert len(x) >= 2 and r, "needs two gfx1100 and one gfx1201"
print(",".join(map(str, x[:2])), r[0])') || { echo "inventory: need two gfx1100 and a gfx1201" >&2; exit 1; }
echo "XTX ranks $XTX, R9700 puller $R9700" | tee -a "$OUT/summary.txt"
rocm-smi --showmeminfo vram --showpids > "$OUT/gpus-before.txt" 2>&1

PULL_PID=""
trap '[ -n "$PULL_PID" ] && kill -INT "$PULL_PID" 2>/dev/null; wait 2>/dev/null' EXIT INT TERM
bench() {
  HIP_VISIBLE_DEVICES="$XTX" ROCR_VISIBLE_DEVICES="$XTX" timeout 1800 "$BIN/r3-arbench" --lib "$LIB" --rows "$ROWS" --iters "$ITERS" --n "$NEMBD" --hc "$HCN"
}
echo "== R9700 idle" | tee -a "$OUT/summary.txt"
bench > "$OUT/pull0.tsv" 2> "$OUT/pull0.err"; echo "rc=$?" >> "$OUT/summary.txt"
column -t -s $'\t' "$OUT/pull0.tsv" | tee -a "$OUT/summary.txt"

echo "== R9700 pulling from host memory" | tee -a "$OUT/summary.txt"
HIP_VISIBLE_DEVICES="$R9700" ROCR_VISIBLE_DEVICES="$R9700" "$BIN/host-pull" > "$OUT/pull1-host-pull.log" 2>&1 &
PULL_PID=$!
sleep 3
bench > "$OUT/pull1.tsv" 2> "$OUT/pull1.err"; echo "rc=$?" >> "$OUT/summary.txt"
kill -INT "$PULL_PID"; wait "$PULL_PID" 2>/dev/null; PULL_PID=""
cat "$OUT/pull1-host-pull.log" >> "$OUT/summary.txt"
column -t -s $'\t' "$OUT/pull1.tsv" | tee -a "$OUT/summary.txt"
rocm-smi --showmeminfo vram --showpids > "$OUT/gpus-after.txt" 2>&1
echo "OUT $OUT"
