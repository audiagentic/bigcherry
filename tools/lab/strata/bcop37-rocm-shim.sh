#!/bin/bash
# BCOP37: the Strata-rocm tree links ROCm by Windows import-library path (${ROCM_INSTALL_DIR}/lib/amdhip64.lib,
# hipblas.lib). Build a shim ROCm root of symlinks to the real install plus those two names pointing at the Linux
# shared objects, so the tree builds on Linux with no source change.
# Usage: bcop37-rocm-shim.sh <real rocm root> <shim dir>
set -eu
R=${1:?real rocm root}; S=${2:?shim dir}
rm -rf "$S"; mkdir -p "$S/lib"
for d in "$R"/*; do n=$(basename "$d"); [ "$n" = lib ] || ln -s "$d" "$S/$n"; done
for f in "$R"/lib/*; do ln -s "$f" "$S/lib/$(basename "$f")"; done
ln -s "$R/lib/libamdhip64.so" "$S/lib/amdhip64.lib"
ln -s "$R/lib/libhipblas.so" "$S/lib/hipblas.lib"
echo "shim $S -> $R"
