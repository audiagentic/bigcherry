#!/bin/bash
# RR01 kernel dev loop: build the kernel candidates under engines/radiance/kernels/kdev/candidates (one small plugin
# each, incremental), run rad-kbench for the given op(s) on one card with every candidate beside the reference, and
# print them side by side (kdev_report.py): correct or not against libref, time per shape, fastest per shape.
# Results are kept per run so two runs can be compared: <out-dir>/kbench.md, kbench.json.
# Run as a queue SCRIPT job so it never overlaps a timed run (one card is enough: VIS=<that card>):
#   VIS=0 SCRIPT kdev-add tools/lab/radiance/kdev.sh @<any build run> <out-dir> add,mul [candidate...]
# The first argument (a llama-server path from the queue) is ignored.
# Usage: kdev.sh <ignored> <out-dir> <op>[,<op>...] [candidate...]      (no candidate = all of them)
# env: RADIANCE_SRC (/mnt/data/bigcherry-work/engines/radiance, built by build.sh), WORK
#      (/mnt/data/bigcherry-work/engines/radiance-rdna3), GPU (HIP index of the card, 0), TARGET (gfx1100),
#      EXTRA_KERNELS (further libraries to run beside the candidates, e.g. libr4d on a gfx12 card),
#      AGAINST (an earlier kbench.json to compare times with), FIXTURE (default: radiance's kernels.rkb), GCC14_BIN
set -u
out=$2 ops=${3:?op[,op...]}; shift 3
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
GCC14_BIN=${GCC14_BIN:-/mnt/data/bigcherry-work/toolchains/gcc14/root/usr/bin}
if command -v g++-14 > /dev/null 2>&1; then
    export CC=gcc-14 CXX=g++-14
else
    export CC="$GCC14_BIN/x86_64-linux-gnu-gcc-14" CXX="$GCC14_BIN/x86_64-linux-gnu-g++-14"
fi
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
[ -x "$src/build/bin/rad-kbench" ] || { echo "NO_RADIANCE_BUILD under $src (run build.sh first)"; exit 1; }

# an out-of-tree plugin builds against an installed radiance; install the existing build once per radiance commit
prefix=$work/radiance-prefix
stamp=$(git -C "$src" rev-parse HEAD)
if [ "$(cat "$prefix/.kdev-commit" 2> /dev/null)" != "$stamp" ]; then
    rm -rf "$prefix"
    cmake --install "$src/build" --prefix "$prefix" > "$out/install.log" 2>&1 || { echo "INSTALL_FAILED"; tail -5 "$out/install.log"; exit 1; }
    echo "$stamp" > "$prefix/.kdev-commit"
fi

build=$work/kdev-$target-build
candidates=$(IFS=';'; echo "$*")
t0=$(date +%s)
if ! cmake -S "$repo/engines/radiance/kernels/kdev" -B "$build" -G Ninja -DCMAKE_PREFIX_PATH="$prefix" \
        -DRADIANCE_SRC="$src" -DKDEV_TARGETS="$target" -DKDEV_CANDIDATES="$candidates" > "$out/configure.log" 2>&1; then
    echo "CONFIGURE_FAILED"; grep -vE "^\s*$" "$out/configure.log" | tail -12 | cut -c1-220; exit 1
fi
if ! cmake --build "$build" -j"$(nproc)" > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error|FAILED" "$out/build.log" | head -20 | cut -c1-240; exit 1
fi
plugins=$(grep -oE "kdev: candidate k_[A-Za-z0-9_]+" "$out/configure.log" | awk '{print $3}' | sort -u | paste -sd, -)
home=$(dirname "$(dirname "$(find "$build" -name 'k_*.so' | head -1)")")
echo "built $plugins for $target in $(( $(date +%s) - t0 )) s"
[ -n "$plugins" ] && [ -d "$home" ] || { echo "NO_PLUGIN built"; exit 1; }

opts=()
for op in ${ops//,/ }; do opts+=(--op "$op"); done
kernels=$plugins${EXTRA_KERNELS:+,$EXTRA_KERNELS},libref
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home:$src/build/radiance_home" timeout "${KBENCH_TIMEOUT:-900}" \
    "$src/build/bin/rad-kbench" --kernels "$kernels" "${opts[@]}" --bench \
    --fixture "${FIXTURE:-$src/build/radiance_home/kernels.rkb}" --report "$out/kbench.md" > "$out/kbench.log" 2>&1
echo "rad-kbench --kernels $kernels ${opts[*]}: exit $?"
grep -E "device:|checked,|memory:|wrote outside|refused|not loaded|ABI|error" "$out/kbench.log" | cut -c1-220
[ -f "$out/kbench.md" ] || { echo "NO_REPORT"; tail -8 "$out/kbench.log" | cut -c1-220; exit 1; }
python3 "$here/kdev_report.py" "$out/kbench.md" --json "$out/kbench.json" ${AGAINST:+--against "$AGAINST"}
echo "report rc=$?"
exit 0
