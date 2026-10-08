#!/bin/bash
# MEN01: build standalone radiance from source on the lab host at a recorded commit, for the cards its kernel library
# covers (gfx1201 = the R9700), and run its tests that need no card. Records commit, toolchain and the plugin list.
# Run as a queue SCRIPT job so the build never overlaps a timed run:
#   VIS=0,1,2,3 SCRIPT radiance-build tools/lab/radiance/build.sh @<any build run> <out-dir>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: build.sh <ignored> <out-dir>
# env: CC / CXX are chosen here (g++-14 if installed, else ROCm clang); RADIANCE_SRC (/mnt/data/bigcherry-work/engines/radiance), RADIANCE_REF (checkout this commit if set),
#      ROCM_PATH (/opt/rocm-7.2.4), RAD_GPU_TARGETS (gfx1201), TESTS (1 = run the card-free ctest set)
set -u
out=$2
mkdir -p "$out"
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
cd "$src" || { echo "NO_SOURCE $src"; exit 1; }
[ -n "${RADIANCE_REF:-}" ] && { git fetch -q origin && git checkout -q "$RADIANCE_REF" || { echo "CHECKOUT_FAILED $RADIANCE_REF"; exit 1; }; }
echo "radiance $(git rev-parse HEAD) $(git log -1 --format=%cd --date=short) ($(git log -1 --format=%s))"
echo "toolchain: $(hipcc --version 2> /dev/null | head -1); cmake $(cmake --version | head -1 | awk '{print $3}'); ROCM_PATH=$ROCM_PATH"
# Upstream builds with g++-14 (its Dockerfile). g++ 13 rejects the compound-literal arrays in abi/rad_builder.h
# ("taking address of temporary array"), so without g++-14 the host compiler is ROCm's clang.
if command -v g++-14 > /dev/null 2>&1; then
    export CC=gcc-14 CXX=g++-14
else
    export CC="$ROCM_PATH/lib/llvm/bin/clang" CXX="$ROCM_PATH/lib/llvm/bin/clang++"
fi
echo "host compiler: $($CXX --version | head -1)"
[ -f build/CMakeCache.txt ] && ! grep -q "CMAKE_CXX_COMPILER:[A-Z]*=$CXX\$" build/CMakeCache.txt && rm -rf build
t0=$(date +%s)
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS="${RAD_GPU_TARGETS:-gfx1201}" > "$out/configure.log" 2>&1 || { echo "CONFIGURE_FAILED"; tail -15 "$out/configure.log"; exit 1; }
grep -iE "gpu target|kernel librar|hip|plugin" "$out/configure.log" | head -12
cmake --build build -j"$(nproc)" > "$out/build.log" 2>&1 || { echo "BUILD_FAILED"; grep -E "error|FAILED" "$out/build.log" | head -15 | cut -c1-240; exit 1; }
echo "build ok in $(( $(date +%s) - t0 )) s; binaries: $(ls build/bin 2> /dev/null | tr '\n' ' ')"
echo "plugins: $(find build/radiance_home -name '*.so' 2> /dev/null | sed 's#.*/##' | sort | tr '\n' ' ')"
if [ "${TESTS:-1}" = 1 ]; then
    ctest --test-dir build -LE 'gpu|oot' > "$out/ctest.log" 2>&1
    echo "card-free tests: $(grep -E "tests passed|tests failed" "$out/ctest.log" | tail -1)"
    grep -E "\*\*\*Failed|\(Failed\)" "$out/ctest.log" | head -8 | cut -c1-160
fi
RADIANCE_HOME=$PWD/build/radiance_home build/bin/radiance --help > "$out/help.txt" 2>&1
echo "serve flags: $(grep -cE "^\s+--" "$out/help.txt") (full text in $out/help.txt)"
exit 0
