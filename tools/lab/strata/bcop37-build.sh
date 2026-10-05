#!/bin/bash
# BCOP37 step 3: clean ROCm build of the pinned Strata-rocm tree for ONE gfx target, following its documented
# commands (docs/USAGE.md section 1). Saves the commands, compiler versions and full logs into the bundle and checks
# the binary links HIP and carries code for the target. Changes no Strata source.
# Usage: bcop37-build.sh <strata tree> <pinned sha> <gfx target> <bundle dir>
set -u
T=${1:?strata tree}; SHA=${2:?sha}; ARCH=${3:?gfx target}; B=${4:?bundle dir}
ROCM=${ROCM:-/mnt/vault/tmp/bc-rocm}
# the tree compiles its .cu files with "-x hip --offload-arch=<arch>", which needs ROCm's clang as the C++ compiler
# (the documented Windows build uses clang-cl); the system g++ rejects those flags
CXX_HIP=$ROCM/lib/llvm/bin/clang++ CC_HIP=$ROCM/lib/llvm/bin/clang
mkdir -p "$B"
cd "$T" || exit 1
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "BUILD_BLOCKED: tree is at $(git rev-parse HEAD), expected $SHA"; exit 1; }
[ -z "$(git status --short | grep -v "^?? build_")" ] || { echo "BUILD_BLOCKED: tree is dirty"; git status --short | head; exit 1; }
{
  echo "repo: $(git remote get-url origin)"; echo "sha: $SHA"; echo "dirty: no"; echo "arch: $ARCH"; echo "ROCM_INSTALL_DIR=$ROCM"
  echo "cmake: $(cmake --version | head -1)"; echo "hip clang: $("$ROCM"/lib/llvm/bin/clang++ --version 2>/dev/null | head -1)"
  echo "configure: cmake -DSTRATA_ENABLE_HIP=ON -DROCM_INSTALL_DIR=$ROCM -DCMAKE_HIP_ARCHITECTURES=$ARCH -DCMAKE_CXX_COMPILER=$CXX_HIP -DCMAKE_C_COMPILER=$CC_HIP -B build_$ARCH"
  echo "build: cmake --build build_$ARCH --target strata -j 16"
} > "$B/strata-version.txt"
export ROCM_INSTALL_DIR=$ROCM PATH="$ROCM/bin:$ROCM/lib/llvm/bin:$PATH" LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
rm -rf "build_$ARCH"
cmake -DSTRATA_ENABLE_HIP=ON -DROCM_INSTALL_DIR="$ROCM" -DCMAKE_HIP_ARCHITECTURES="$ARCH" -DCMAKE_CXX_COMPILER="$CXX_HIP" -DCMAKE_C_COMPILER="$CC_HIP" -B "build_$ARCH" > "$B/configure.$ARCH.log" 2>&1
echo "configure rc=$?"; tail -4 "$B/configure.$ARCH.log"
cmake --build "build_$ARCH" --target strata -j 16 > "$B/build.$ARCH.log" 2>&1
rc=$?; echo "build rc=$rc"
if [ $rc -ne 0 ]; then grep -nE "error:|fatal error|undefined reference|\*\*\*" "$B/build.$ARCH.log" | head -25 | cut -c1-300; echo BUILD_FAILED; exit 1; fi
bin=$(find "build_$ARCH" -maxdepth 3 -type f -name strata | head -1)
echo "binary: $bin"
ldd "$bin" | grep -iE "amdhip|hip|roc" | tee -a "$B/strata-version.txt"
echo "target code objects: $(strings -a "$bin" | grep -c "$ARCH")" | tee -a "$B/strata-version.txt"
echo BUILD_DONE
