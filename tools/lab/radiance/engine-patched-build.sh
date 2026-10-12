#!/bin/bash
# Build radiance 1.3.0 with BigCherry's engine patches (engines/radiance/engine-patches/*.patch) in its own source tree,
# beside the pinned unpatched one, so a lab run can use either: RADIANCE_SRC=<this tree> for the patched engine.
# Usage: engine-patched-build.sh   env: SRC (pinned tree), OUT (patched tree)
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
src=${SRC:-/mnt/data/bigcherry-work/engines/radiance}
out=${OUT:-/mnt/data/bigcherry-work/engines/radiance-patched}
pin=$(git -C "$src" rev-parse HEAD)
if [ ! -d "$out/.git" ] && [ ! -f "$out/.git" ]; then git -C "$src" worktree add --detach "$out" "$pin" > /dev/null; fi
git -C "$out" checkout -q --detach "$pin"
git -C "$out" reset -q --hard "$pin"
git -C "$out" clean -fdq   # files a patch created last time
for p in engines/radiance/engine-patches/*.patch; do
    git -C "$out" apply --recount "$PWD/$p" || { echo "ENGINE_PATCH_STOP $(basename "$p") does not apply"; exit 1; }
    echo "applied $(basename "$p")"
done
args=$(grep -E "^(CMAKE_HIP_ARCHITECTURES|RAD_GPU_TARGETS|CMAKE_PREFIX_PATH|CMAKE_HIP_COMPILER|CMAKE_CXX_COMPILER|CMAKE_C_COMPILER):" "$src/build/CMakeCache.txt" | sed -E 's/^([A-Z_]+):[A-Z]+=(.*)$/-D\1=\2/')
cmake -S "$out" -B "$out/build" -G Ninja $args > "$out/configure.log" 2>&1 || { echo "ENGINE_PATCH_STOP configure"; tail -5 "$out/configure.log"; exit 1; }
# the engine and its tools; the unit tests are not needed to serve and arch_test does not link the patched core
cmake --build "$out/build" -j20 --target radiance rad-info rad-kbench rad-tune > "$out/build.log" 2>&1 || { echo "ENGINE_PATCH_STOP build"; grep -E "error" "$out/build.log" | head -5; exit 1; }
echo "ENGINE_PATCHED_OK $out/build/bin/radiance"
